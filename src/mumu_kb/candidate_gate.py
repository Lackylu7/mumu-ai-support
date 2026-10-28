from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class CandidatePackError(ValueError):
    pass


@dataclass(frozen=True)
class CandidateGateResult:
    candidate_id: str
    intent: str
    ready: bool
    reasons: tuple[str, ...]
    requires_interpretation: bool


@dataclass(frozen=True)
class CandidateGateReport:
    results: tuple[CandidateGateResult, ...]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def ready_count(self) -> int:
        return sum(result.ready for result in self.results)

    @property
    def blocked_count(self) -> int:
        return self.total - self.ready_count

    @property
    def interpretation_candidate_count(self) -> int:
        return sum(result.requires_interpretation for result in self.results)

    def to_markdown(self) -> str:
        lines = [
            "# 候选知识安全发布闸门报告",
            "",
            "> 本报告只做本地校验，不会写入木木正式知识表。",
            "",
            "## 汇总",
            "",
            f"- 候选总数：{self.total}",
            f"- 可发布：{self.ready_count}",
            f"- 被阻塞：{self.blocked_count}",
            f"- 建议生成制度解读：{self.interpretation_candidate_count}",
            "",
            "## 明细",
            "",
            "| 候选编号 | 用户意图 | 闸门结果 | 原因 |",
            "|---|---|---|---|",
        ]
        for result in self.results:
            status = "可发布" if result.ready else "已阻塞"
            reasons = "；".join(result.reasons) if result.reasons else "无阻塞项"
            lines.append(
                f"| {result.candidate_id} | {result.intent} | {status} | {reasons} |"
            )
        lines.append("")
        return "\n".join(lines)


def _text(candidate: dict[str, Any], field: str) -> str:
    value = candidate.get(field)
    return value.strip() if isinstance(value, str) else ""


def _strings(candidate: dict[str, Any], field: str) -> tuple[str, ...]:
    value = candidate.get(field)
    if not isinstance(value, list):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())


def evaluate_candidate(candidate: dict[str, Any]) -> CandidateGateResult:
    reasons: list[str] = []
    status = _text(candidate, "candidate_status")
    blockers = _strings(candidate, "blockers")

    if status != "可发布":
        reasons.append("候选状态不是“可发布”")
    reasons.extend(blockers)
    if not _text(candidate, "source_title"):
        reasons.append("缺少来源标题")
    if not _text(candidate, "source_node_id"):
        reasons.append("缺少来源节点ID")
    if not _text(candidate, "source_section"):
        reasons.append("缺少来源章节")
    if not _text(candidate, "effective_from"):
        reasons.append("缺少明确生效时间")
    if not _text(candidate, "canonical_answer"):
        reasons.append("缺少标准答案")
    if not _strings(candidate, "question_variants"):
        reasons.append("缺少常见问法")

    requires_interpretation = candidate.get("requires_interpretation") is True
    if requires_interpretation and not _strings(candidate, "interpretation_topics"):
        reasons.append("标记需要制度解读，但缺少解读主题")

    return CandidateGateResult(
        candidate_id=_text(candidate, "candidate_id") or "未编号",
        intent=_text(candidate, "intent") or "未填写意图",
        ready=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        requires_interpretation=requires_interpretation,
    )


def load_candidate_pack(path: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidatePackError(f"无法读取候选知识包: {path}") from error
    if not isinstance(raw, list):
        raise CandidatePackError("候选知识包顶层必须是数组")

    candidates: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise CandidatePackError(f"第{index}条候选必须是对象")
        candidate_id = _text(item, "candidate_id")
        if not candidate_id:
            raise CandidatePackError(f"第{index}条候选缺少编号")
        if candidate_id in candidate_ids:
            raise CandidatePackError(f"发现重复候选编号: {candidate_id}")
        if item.get("excluded_from_runtime") is not True:
            raise CandidatePackError(f"候选 {candidate_id} 必须排除在运行时之外")
        candidate_ids.add(candidate_id)
        candidates.append(item)
    return candidates


def build_gate_report(candidates: Iterable[dict[str, Any]]) -> CandidateGateReport:
    return CandidateGateReport(tuple(evaluate_candidate(item) for item in candidates))


def write_gate_report(candidate_path: Path, report_path: Path) -> CandidateGateReport:
    report = build_gate_report(load_candidate_pack(candidate_path))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.to_markdown(), encoding="utf-8")
    return report
