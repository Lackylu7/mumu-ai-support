from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, TextIO

from .dws import DwsClient
from .case_migration import apply_case_migration_plan, build_case_migration_plan
from .interpretation_importer import load_interpretation_pack, sync_interpretations
from .interpretation_migration import (
    apply_interpretation_migration_plan,
    build_interpretation_migration_plan,
)
from .importer import (
    apply_import_plan,
    build_import_plan,
    expected_ids_from_pack,
    verify_remote_records,
    write_plan_files,
)
from .models import KnowledgeEntry, ReviewIssue, load_json_records
from .provision import apply_provision_plan, build_provision_plan, load_manifest
from .review import render_review_packet


DEFAULT_PACK = Path("knowledge/employee-partnership-3.0")
DEFAULT_MANIFEST = Path(".runtime/aitable-manifest.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mumu-kb")
    subparsers = parser.add_subparsers(dest="command", required=True)

    provision = subparsers.add_parser("provision", help="规划或创建木木知识库")
    provision.add_argument("--profile", required=True)
    provision.add_argument("--apply", action="store_true")
    provision.add_argument("--reuse-base-id")

    import_parser = subparsers.add_parser("import", help="规划或写入首份知识包")
    import_parser.add_argument("--profile", required=True)
    import_parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    import_parser.add_argument("--apply", action="store_true")

    verify = subparsers.add_parser("verify", help="全量回读校验")
    verify.add_argument("--profile", required=True)
    verify.add_argument("--pack", type=Path, default=DEFAULT_PACK)

    review = subparsers.add_parser("review", help="生成人工知识审核清单")
    review.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    review.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PACK / "review-packet.md",
    )
    migrate_cases = subparsers.add_parser(
        "migrate-cases",
        help="预览或迁移木木自动答疑群字段",
    )
    migrate_cases.add_argument("--profile", required=True)
    migrate_cases.add_argument("--apply", action="store_true")
    migrate_interpretations = subparsers.add_parser(
        "migrate-interpretations",
        help="预览或创建制度解读数据表",
    )
    migrate_interpretations.add_argument("--profile", required=True)
    migrate_interpretations.add_argument("--apply", action="store_true")
    sync_parser = subparsers.add_parser(
        "sync-interpretations",
        help="创建或更新制度解读记录",
    )
    sync_parser.add_argument("--profile", required=True)
    sync_parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_PACK / "interpretations.json",
    )
    sync_parser.add_argument("--apply", action="store_true")
    return parser


def _confirm(input_func: Callable[[str], str], phrase: str) -> bool:
    return input_func(f"请输入“{phrase}”确认：").strip() == phrase


def main(
    argv: list[str] | None = None,
    *,
    client_factory: Callable[[str], object] = DwsClient,
    input_func: Callable[[str], str] = input,
    stdout: TextIO = sys.stdout,
) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "review":
            entries = load_json_records(args.pack / "entries.json", KnowledgeEntry)
            issues = load_json_records(args.pack / "review-issues.json", ReviewIssue)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                render_review_packet(entries, issues),
                encoding="utf-8",
            )
            print(f"已生成审核清单：{args.output}（{len(entries)} 条）", file=stdout)
            return 0

        client = client_factory(args.profile)
        if args.command == "migrate-interpretations":
            manifest = load_manifest(DEFAULT_MANIFEST)
            actions = build_interpretation_migration_plan(client, manifest)
            print(f"制度解读迁移预览：{len(actions)} 个动作。", file=stdout)
            for action in actions:
                print(f"- {action.kind}: 制度解读", file=stdout)
            if not args.apply:
                return 0
            if not _confirm(input_func, "迁移木木制度解读"):
                print("确认短语不匹配，未执行任何写入。", file=stdout)
                return 2
            updated = apply_interpretation_migration_plan(
                client,
                manifest,
                actions,
                DEFAULT_MANIFEST,
            )
            fields = updated["tables"]["制度解读"]["fields"]
            print(f"迁移及回读完成，制度解读字段数={len(fields)}。", file=stdout)
            return 0

        if args.command == "sync-interpretations":
            entries = load_interpretation_pack(
                args.file,
                DEFAULT_PACK / "entries.json",
            )
            print(f"制度解读同步预览：{len(entries)} 条。", file=stdout)
            if not args.apply:
                return 0
            if not _confirm(input_func, "写入木木制度解读"):
                print("确认短语不匹配，未执行任何写入。", file=stdout)
                return 2
            summary = sync_interpretations(
                client,
                load_manifest(DEFAULT_MANIFEST),
                entries,
                confirm=True,
            )
            print(
                f"制度解读同步完成：新增{summary.created}，更新{summary.updated}，"
                f"远端总数{summary.remote_count}。",
                file=stdout,
            )
            return 0

        if args.command == "migrate-cases":
            manifest = load_manifest(DEFAULT_MANIFEST)
            actions = build_case_migration_plan(client, manifest)
            print(f"案件字段迁移预览：{len(actions)} 个动作。", file=stdout)
            for action in actions:
                print(
                    f"- {action.kind}: {action.table}.{action.field}",
                    file=stdout,
                )
            if not args.apply:
                return 0
            if not _confirm(input_func, "迁移木木案件字段"):
                print("确认短语不匹配，未执行任何写入。", file=stdout)
                return 2
            updated = apply_case_migration_plan(
                client,
                manifest,
                actions,
                DEFAULT_MANIFEST,
            )
            gap_fields = updated["tables"]["知识缺口"]["fields"]
            print(
                f"迁移及回读完成，知识缺口字段数={len(gap_fields)}。",
                file=stdout,
            )
            return 0

        if args.command == "provision":
            actions = build_provision_plan(
                client,
                reuse_base_id=args.reuse_base_id,
            )
            print(f"Provision 计划：{len(actions)} 个动作，目标 Base=木木知识库", file=stdout)
            for action in actions:
                print(f"- {action.kind}: {action.table or '木木知识库'}", file=stdout)
            if not args.apply:
                return 0
            if not _confirm(input_func, "创建木木知识库"):
                print("确认短语不匹配，未执行任何写入。", file=stdout)
                return 2
            manifest = apply_provision_plan(client, actions, DEFAULT_MANIFEST)
            print(
                f"已创建 Base，baseId={manifest['base_id']}，数据表={len(manifest['tables'])}",
                file=stdout,
            )
            return 0

        if args.command == "import":
            batches = build_import_plan(args.pack, DEFAULT_MANIFEST, datetime.now().astimezone())
            plan_files = write_plan_files(batches, Path(".runtime"))
            record_count = sum(len(batch.records) for batch in batches)
            print(f"导入计划：{len(plan_files)} 批，{record_count} 条；尚未写入。", file=stdout)
            if not args.apply:
                return 0
            if not _confirm(input_func, "写入木木知识库"):
                print("确认短语不匹配，未执行任何写入。", file=stdout)
                return 2
            summary = apply_import_plan(client, plan_files, confirm=True)
            print(
                f"已写入 {summary.completed_batches} 批、{summary.record_count} 条。",
                file=stdout,
            )
            return 0

        manifest = load_manifest(DEFAULT_MANIFEST)
        expected = expected_ids_from_pack(args.pack)
        summary = verify_remote_records(client, manifest, expected)
        print(f"回读业务记录数：{summary.table_counts}", file=stdout)
        if any(summary.empty_rows.values()):
            print(f"系统空白行：{summary.empty_rows}", file=stdout)
        if not summary.ok:
            print(
                f"校验失败，缺失={summary.missing_ids}，重复={summary.duplicate_ids}",
                file=stdout,
            )
            return 3
        print("全量回读校验通过。", file=stdout)
        return 0
    except Exception as error:
        print(f"错误：{error}", file=stdout)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
