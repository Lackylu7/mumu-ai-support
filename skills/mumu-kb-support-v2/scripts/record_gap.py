#!/usr/bin/env python3
"""record_gap.py - 木木知识缺口记录生成器（实时检索型技能配套）。

用法:
  python3 record_gap.py --question "新店分红从什么时候起算？" --intent 分红结算 --domain 员工合伙制 --topic 分红与收入
  python3 record_gap.py --question "..." --intent 兜底 --domain 公司通用 --topic "" --asker "张三" --note "群聊缺少必要信息" --dry-run

行为:
  - 根据未命中问题生成「知识缺口」表记录（JSON），供 Agent 用 create_records 实时写入钉钉 AI 表格
  - 自动生成 MUMU-GAP-<时间戳> 格式的自动来源编号（与表内现有记录一致）
  - 自动填首次出现/最近出现时间、出现次数=1、处理状态=待处理
  - --dry-run 仅打印预览，不输出最终 JSON（用于检查）
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta

# 知识缺口表字段映射（字段名 -> 字段ID），与木木知识库 LbNxnF4 表一致
GAP_TABLE_ID = "LbNxnF4"
FIELDS = {
    "问题摘要": "2Ok5T1R",        # text
    "案件编号": "imk9JPY",        # richText
    "一级业务域": "gjZOmRP",      # singleSelect
    "出现次数": "b7znPsf",        # number
    "首次出现": "PjlXuVI",        # date
    "最近出现": "EBCjvsv",        # date
    "处理状态": "bYbmLxA",        # singleSelect
    "当前负责人": "j0qQlZw",      # text
    "补充知识编号": "F0w7Twd",    # text
    "处理备注": "G5DsAOl",        # richText
    "自动来源编号": "gHBo14z",    # text
    "二级主题": "9YsZviO",        # singleSelect
}

# 合法取值（与表内单选选项一致）
DOMAINS = ["内部政策示例", "公司通用", "教练运营", "人力与行政", "员工合伙制", "门店运营", "客户服务与品牌"]
STATUS_PENDING = "待处理"


def make_gap_number(question: str) -> str:
    """生成 MUMU-GAP-<hash> 编号，与表内 MUMU-GAP-20260813-XXXX 风格一致。"""
    digest = hashlib.sha1(question.strip().encode("utf-8")).hexdigest()[:10].upper()
    today = datetime.now().strftime("%Y%m%d")
    return f"MUMU-GAP-{today}-{digest}"


def build_record(question: str, intent: str = "", domain: str = "公司通用",
                 topic: str = "", asker: str = "", note: str = "") -> dict:
    """组装知识缺口表记录（create_records 入参）。"""
    if domain not in DOMAINS:
        domain = "公司通用"
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+08:00")

    cells = {
        FIELDS["问题摘要"]: question,
        FIELDS["一级业务域"]: domain,
        FIELDS["出现次数"]: 1,
        FIELDS["首次出现"]: now,
        FIELDS["最近出现"]: now,
        FIELDS["处理状态"]: STATUS_PENDING,
        FIELDS["当前负责人"]: asker or "待指派",
        FIELDS["自动来源编号"]: make_gap_number(question),
    }
    if topic:
        cells[FIELDS["二级主题"]] = topic
    if intent:
        note_full = f"意图：{intent}" + (f"；备注：{note}" if note else "")
        cells[FIELDS["处理备注"]] = note_full
    elif note:
        cells[FIELDS["处理备注"]] = note

    return {
        "baseId": "3NwLYZXWynKpDxYvsNAzk5GZVkyEqBQm",
        "tableId": GAP_TABLE_ID,
        "records": [{"fields": cells}],
    }


def main():
    parser = argparse.ArgumentParser(description="生成木木知识缺口表记录")
    parser.add_argument("--question", required=True, help="未命中的问题原文")
    parser.add_argument("--intent", default="", help="意图分类（如: 分红结算/兜底）")
    parser.add_argument("--domain", default="公司通用", help="一级业务域（默认公司通用）")
    parser.add_argument("--topic", default="", help="二级主题（可空）")
    parser.add_argument("--asker", default="", help="提问人姓名（可空）")
    parser.add_argument("--note", default="", help="补充说明（可空）")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不输出最终 JSON")
    args = parser.parse_args()

    record = build_record(args.question, args.intent, args.domain, args.topic, args.asker, args.note)

    if args.dry_run:
        print("[DRY-RUN] 将写入知识缺口表:")
        print(f"  问题摘要: {args.question}")
        print(f"  意图/领域/主题: {args.intent or '-'} / {args.domain} / {args.topic or '-'}")
        print(f"  自动来源编号: {record['records'][0]['fields'][FIELDS['自动来源编号']]}")
        print(f"  处理状态: {STATUS_PENDING}")
        return

    print(json.dumps(record, ensure_ascii=False, indent=2))
    print("\n# 下一步：用「钉钉 AI 表格」MCP 的 create_records 工具，传入上方 JSON 写入知识缺口表",
          file=sys.stderr)


if __name__ == "__main__":
    main()
