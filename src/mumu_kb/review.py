from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .models import KnowledgeEntry, ReviewIssue, ReviewIssueStatus


def _stable_number(value: str) -> tuple[str, int]:
    prefix, _, suffix = value.rpartition("K")
    try:
        return prefix, int(suffix)
    except ValueError:
        return value, 0


def _line_list(label: str, values: Iterable[str]) -> list[str]:
    items = [value.strip() for value in values if value.strip()]
    return [f"- {label}：{'；'.join(items) if items else '无'}"]


def render_review_packet(
    entries: list[KnowledgeEntry],
    issues: list[ReviewIssue],
) -> str:
    pending_issues = [
        issue for issue in issues if issue.status is not ReviewIssueStatus.RESOLVED
    ]
    resolved_issues = [
        issue for issue in issues if issue.status is ReviewIssueStatus.RESOLVED
    ]
    blocked_ids = {
        knowledge_id
        for issue in pending_issues
        for knowledge_id in issue.blocked_knowledge_ids
    }
    sections: dict[str, list[KnowledgeEntry]] = defaultdict(list)
    for entry in sorted(entries, key=lambda item: _stable_number(item.knowledge_id)):
        sections[entry.source_section].append(entry)

    lines = [
        "# 木木知识库首批人工审核清单",
        "",
        f"- 知识条目总数：{len(entries)}",
        f"- 明确阻塞问题：{len(pending_issues)}",
        f"- 已解决审核问题：{len(resolved_issues)}",
        "- 审核原则：只确认原文是否被准确拆解，不补充原文没有的规则。",
        "",
        "## 待确认阻塞问题",
        "",
    ]
    if not pending_issues:
        lines.append("无。")
    for issue in pending_issues:
        blocked = "、".join(issue.blocked_knowledge_ids) or "无"
        lines.extend(
            [
                f"### {issue.review_id} · {issue.issue_type}",
                "",
                f"- 原文：{issue.quoted_text}",
                f"- 章节：{issue.source_section}",
                f"- 需要确认：{issue.question_for_owner}",
                f"- 当前状态：{issue.status.value}",
                f"- 阻塞知识：{blocked}",
                "",
            ]
        )

    lines.extend(["", "## 已解决审核问题", ""])
    if not resolved_issues:
        lines.append("无。")
    for issue in resolved_issues:
        related = "、".join(issue.blocked_knowledge_ids) or "无"
        lines.extend(
            [
                f"### {issue.review_id} · {issue.issue_type}",
                "",
                f"- 原文：{issue.quoted_text}",
                f"- 章节：{issue.source_section}",
                f"- 处理结论：{issue.question_for_owner}",
                f"- 当前状态：{issue.status.value}",
                f"- 关联知识：{related}",
                "",
            ]
        )

    lines.extend(["## 知识条目", ""])
    for section, section_entries in sections.items():
        lines.extend([f"## {section}", ""])
        for entry in section_entries:
            blocked = "（阻塞）" if entry.knowledge_id in blocked_ids else ""
            lines.extend(
                [
                    f"### {entry.knowledge_id}{blocked} · {entry.intent}",
                    "",
                    f"- 标准答案：{entry.canonical_answer}",
                    f"- 风险等级：{entry.risk_level.value}",
                    f"- 敏感级别：{entry.sensitivity.value}",
                    f"- 状态：{entry.status.value}",
                    f"- 来源：{entry.source_id} / {entry.source_section} / v{entry.version}",
                    *_line_list("成立条件", entry.conditions),
                    *_line_list("例外情形", entry.exceptions),
                    *_line_list("办理步骤", entry.steps),
                    "- 审核结论：[ ] 通过　[ ] 需修改　[ ] 暂不发布",
                    "- 审核备注：",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"
