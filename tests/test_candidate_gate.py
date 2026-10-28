import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mumu_kb.candidate_gate import (
    CandidatePackError,
    build_gate_report,
    evaluate_candidate,
    load_candidate_pack,
)


def candidate(**overrides):
    value = {
        "candidate_id": "COO-C001",
        "business_domain": "教练运营/教练晋升与分级",
        "intent": "晋升OA提交人",
        "question_variants": ["教练晋升OA谁提交？"],
        "canonical_answer": "晋升OA由教练本人提交。",
        "conditions": [],
        "exceptions": [],
        "steps": [],
        "source_title": "教练分级介绍及操作SOP",
        "source_node_id": "node-1",
        "source_section": "常见问题",
        "effective_from": "2026-08-12T00:00:00+08:00",
        "effective_to": "",
        "risk_level": "普通",
        "candidate_status": "可发布",
        "blockers": [],
        "requires_interpretation": False,
        "interpretation_topics": [],
        "excluded_from_runtime": True,
    }
    value.update(overrides)
    return value


class CandidateGateTests(unittest.TestCase):
    def test_publishable_candidate_passes_without_leaving_intake_isolation(self) -> None:
        result = evaluate_candidate(candidate())

        self.assertTrue(result.ready)
        self.assertEqual(result.reasons, ())

    def test_conflict_and_existing_blockers_prevent_publication(self) -> None:
        result = evaluate_candidate(
            candidate(
                candidate_status="存在冲突",
                blockers=["3万元规则与2万元示例直接冲突"],
            )
        )

        self.assertFalse(result.ready)
        self.assertIn("候选状态不是“可发布”", result.reasons)
        self.assertIn("3万元规则与2万元示例直接冲突", result.reasons)

    def test_publishable_candidate_requires_source_and_effective_time(self) -> None:
        result = evaluate_candidate(
            candidate(source_node_id="", effective_from="")
        )

        self.assertFalse(result.ready)
        self.assertIn("缺少来源节点ID", result.reasons)
        self.assertIn("缺少明确生效时间", result.reasons)

    def test_candidate_pack_rejects_runtime_enabled_records(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.json"
            path.write_text(
                json.dumps([candidate(excluded_from_runtime=False)], ensure_ascii=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(CandidatePackError, "必须排除在运行时之外"):
                load_candidate_pack(path)

    def test_report_counts_ready_blocked_and_interpretation_candidates(self) -> None:
        report = build_gate_report(
            [
                candidate(),
                candidate(
                    candidate_id="COO-C002",
                    candidate_status="待负责人确认",
                    blockers=["需确认现行版本"],
                    requires_interpretation=True,
                    interpretation_topics=["解释制度目的"],
                ),
            ]
        )

        self.assertEqual(report.total, 2)
        self.assertEqual(report.ready_count, 1)
        self.assertEqual(report.blocked_count, 1)
        self.assertEqual(report.interpretation_candidate_count, 1)
        self.assertIn("COO-C002", report.to_markdown())
        self.assertIn("需确认现行版本", report.to_markdown())


if __name__ == "__main__":
    unittest.main()
