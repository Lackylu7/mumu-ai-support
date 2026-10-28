from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from mumu_kb.models import ExpectedAction, RiskLevel, TestCase, TestCategory

from .domain import AnswerResult


class EvaluationService(Protocol):
    def answer(
        self,
        user_id: str,
        question: str,
        now: datetime,
    ) -> AnswerResult: ...


class PointJudge(Protocol):
    def judge_points(
        self,
        question: str,
        answer: str,
        expected_points: Sequence[str],
    ) -> tuple[str, ...]: ...


class NullGapRecorder:
    def record(self, question: str, reason: str, now: datetime) -> None:
        del question, reason, now


def _normalized(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    return re.sub(r"[^0-9a-z%\u4e00-\u9fff]+", "", normalized)


def _point_matches(point: str, answer: str) -> bool:
    normalized_point = _normalized(point)
    normalized_answer = _normalized(answer)
    if not normalized_point:
        return False
    if normalized_point in normalized_answer:
        return True
    numeric_facts = re.findall(
        r"\d+(?:\.\d+)?(?:万|千|百)?(?:元|%|折|天|日|月|年|家|店|小时|分钟|km)?",
        unicodedata.normalize("NFKC", point).lower(),
    )
    return bool(numeric_facts) and all(
        _normalized(fact) in normalized_answer for fact in numeric_facts
    )


@dataclass(frozen=True)
class CaseEvaluation:
    test_id: str
    category: str
    risk_level: str
    question: str
    expected_action: str
    actual_action: str
    passed: bool
    action_correct: bool
    source_correct: bool
    missing_points: tuple[str, ...]
    expected_knowledge_ids: tuple[str, ...]
    actual_knowledge_ids: tuple[str, ...]
    answer: str
    handoff_reason: str
    error_type: str = ""


@dataclass(frozen=True)
class EvaluationReport:
    generated_at: str
    total_cases: int
    passed_cases: int
    knowledge_point_accuracy: float
    source_match_rate: float
    unsupported_assertion_rate: float
    handoff_accuracy: float
    action_accuracy: float
    high_risk_fact_errors: int
    passes_gate: bool
    cases: tuple[CaseEvaluation, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_markdown(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        status = "通过" if self.passes_gate else "未通过"
        lines = [
            "# 木木准确率验收报告",
            "",
            f"- 验收结论：{status}",
            f"- 测试题：{self.total_cases}",
            f"- 完全通过：{self.passed_cases}",
            f"- 知识要点准确率：{self.knowledge_point_accuracy:.2%}",
            f"- 来源匹配率：{self.source_match_rate:.2%}",
            f"- 无依据业务断言率：{self.unsupported_assertion_rate:.2%}",
            f"- 应转人工正确率：{self.handoff_accuracy:.2%}",
            f"- 动作准确率：{self.action_accuracy:.2%}",
            f"- 高风险事实错误：{self.high_risk_fact_errors}",
            "",
            "## 未通过项目",
            "",
        ]
        failures = [item for item in self.cases if not item.passed]
        if not failures:
            lines.append("无。")
        for item in failures:
            details: list[str] = []
            if not item.action_correct:
                details.append(
                    f"动作应为{item.expected_action}，实际为{item.actual_action}"
                )
            if not item.source_correct:
                details.append(
                    "来源不完整："
                    + "、".join(item.expected_knowledge_ids)
                )
            if item.missing_points:
                details.append("缺少要点：" + "；".join(item.missing_points))
            if item.error_type:
                details.append("运行错误：" + item.error_type)
            lines.extend(
                [
                    f"### {item.test_id} {item.question}",
                    "",
                    "- " + "；".join(details),
                    f"- 实际依据：{'、'.join(item.actual_knowledge_ids) or '无'}",
                    f"- 转人工原因：{item.handoff_reason or '无'}",
                    f"- 实际输出：{item.answer}",
                    "",
                ]
            )
        path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_cases(
    service: EvaluationService,
    test_cases: Sequence[TestCase],
    now: datetime,
    point_judge: PointJudge | None = None,
) -> EvaluationReport:
    results: list[CaseEvaluation] = []
    expected_point_count = 0
    matched_point_count = 0
    expected_source_count = 0
    matched_source_count = 0
    expected_handoff_count = 0
    correct_handoff_count = 0
    answered_count = 0
    unsupported_answer_count = 0
    correct_action_count = 0
    high_risk_fact_errors = 0

    for test_case in test_cases:
        session_id = f"offline-evaluation:{test_case.test_id}"
        prompt = "\n".join((*test_case.context, test_case.question))
        error_type = ""
        try:
            if test_case.category is TestCategory.CONTEXT_FOLLOWUP:
                for context_turn in test_case.context:
                    service.answer(session_id, context_turn, now)
                prompt = test_case.question
            answer_result = service.answer(
                session_id,
                prompt,
                now,
            )
        except Exception as error:
            answer_result = AnswerResult(text="系统评估调用失败。")
            error_type = type(error).__name__

        expected_answer = test_case.expected_action is ExpectedAction.ANSWER
        actual_handoff = answer_result.is_handoff
        actual_answer = not actual_handoff and not error_type
        actual_action = "转人工" if actual_handoff else "回答"
        if error_type:
            actual_action = "系统错误"
        action_correct = (
            expected_answer and actual_answer
        ) or (
            not expected_answer and actual_handoff
        )
        correct_action_count += int(action_correct)

        actual_ids = tuple(item.knowledge_id for item in answer_result.evidence)
        actual_id_set = set(actual_ids)
        expected_ids = tuple(test_case.knowledge_ids)
        expected_source_count += len(expected_ids) if expected_answer else 0
        matched_source_count += (
            len(set(expected_ids) & actual_id_set) if expected_answer else 0
        )
        source_correct = not expected_answer or set(expected_ids).issubset(actual_id_set)

        missing_points: tuple[str, ...] = ()
        if expected_answer:
            expected_point_count += len(test_case.expected_points)
            deterministic_matches = tuple(
                point
                for point in test_case.expected_points
                if actual_answer and _point_matches(point, answer_result.text)
            )
            semantic_matches: tuple[str, ...] = ()
            deterministic_missing = tuple(
                point
                for point in test_case.expected_points
                if point not in deterministic_matches
            )
            if actual_answer and deterministic_missing and point_judge is not None:
                matched_semantically: list[str] = []
                for _ in range(2):
                    remaining = tuple(
                        point
                        for point in deterministic_missing
                        if point not in matched_semantically
                    )
                    if not remaining:
                        break
                    try:
                        judged = point_judge.judge_points(
                            prompt,
                            answer_result.text,
                            remaining,
                        )
                    except Exception:
                        # A judge outage must not turn an evaluation run into a bot outage.
                        continue
                    matched_semantically.extend(
                        point
                        for point in remaining
                        if point in judged and point not in matched_semantically
                    )
                semantic_matches = tuple(matched_semantically)
            matched_points = deterministic_matches + semantic_matches
            matched_point_count += len(matched_points)
            missing_points = tuple(
                point
                for point in test_case.expected_points
                if point not in matched_points
            )
        else:
            expected_handoff_count += 1
            correct_handoff_count += int(actual_handoff)

        allows_answer_without_evidence = (
            test_case.category is TestCategory.GENERAL_CHAT
        )
        if actual_answer:
            answered_count += 1
            if not expected_answer or (
                not actual_ids and not allows_answer_without_evidence
            ):
                unsupported_answer_count += 1

        fact_error = (
            test_case.risk_level is RiskLevel.HIGH
            and actual_answer
            and (not expected_answer or bool(missing_points))
        )
        high_risk_fact_errors += int(fact_error)
        passed = (
            not error_type
            and action_correct
            and source_correct
            and not missing_points
        )
        results.append(
            CaseEvaluation(
                test_id=test_case.test_id,
                category=test_case.category.value,
                risk_level=test_case.risk_level.value,
                question=test_case.question,
                expected_action=test_case.expected_action.value,
                actual_action=actual_action,
                passed=passed,
                action_correct=action_correct,
                source_correct=source_correct,
                missing_points=missing_points,
                expected_knowledge_ids=expected_ids,
                actual_knowledge_ids=actual_ids,
                answer=answer_result.text,
                handoff_reason=(
                    answer_result.handoff.reason
                    if answer_result.handoff is not None
                    else ""
                ),
                error_type=error_type,
            )
        )

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 1.0

    knowledge_point_accuracy = rate(matched_point_count, expected_point_count)
    source_match_rate = rate(matched_source_count, expected_source_count)
    unsupported_assertion_rate = rate(unsupported_answer_count, answered_count)
    handoff_accuracy = rate(correct_handoff_count, expected_handoff_count)
    action_accuracy = rate(correct_action_count, len(test_cases))
    passes_gate = (
        high_risk_fact_errors == 0
        and knowledge_point_accuracy >= 0.95
        and source_match_rate >= 0.98
        and unsupported_assertion_rate <= 0.01
        and handoff_accuracy >= 0.95
    )
    return EvaluationReport(
        generated_at=now.isoformat(),
        total_cases=len(test_cases),
        passed_cases=sum(item.passed for item in results),
        knowledge_point_accuracy=knowledge_point_accuracy,
        source_match_rate=source_match_rate,
        unsupported_assertion_rate=unsupported_assertion_rate,
        handoff_accuracy=handoff_accuracy,
        action_accuracy=action_accuracy,
        high_risk_fact_errors=high_risk_fact_errors,
        passes_gate=passes_gate,
        cases=tuple(results),
    )
