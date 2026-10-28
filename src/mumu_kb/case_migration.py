from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .provision import _write_manifest
from .schema import BUSINESS_DOMAINS, FieldSpec, get_table_spec


class CaseMigrationError(RuntimeError):
    pass


CASE_GAP_FIELDS = (
    "原消息ID",
    "提问人ID",
    "提问人姓名",
    "案件群ID",
    "案件群名",
    "建群时间",
    "结案触发人ID",
    "结案触发人姓名",
    "结案时间",
    "自动来源编号",
    "负责人答复",
    "提交沉淀",
)


@dataclass(frozen=True)
class CaseMigrationAction:
    kind: str
    table: str
    field: str
    payload: dict[str, object]


def _identifier(item: dict[str, object], *names: str) -> str:
    for name in names:
        value = item.get(name)
        if isinstance(value, str) and value:
            return value
    return ""


def _field_name(item: dict[str, object]) -> str:
    return _identifier(item, "fieldName", "name")


def _table_fields(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        return []
    tables = data.get("tables")
    if isinstance(tables, list) and tables and isinstance(tables[0], dict):
        fields = tables[0].get("fields")
        if isinstance(fields, list):
            return [item for item in fields if isinstance(item, dict)]
    fields = data.get("fields")
    if isinstance(fields, list):
        return [item for item in fields if isinstance(item, dict)]
    return []


def _table_info(manifest: dict[str, object], table_name: str) -> tuple[str, str]:
    base_id = manifest.get("base_id")
    tables = manifest.get("tables")
    if not isinstance(base_id, str) or not isinstance(tables, dict):
        raise CaseMigrationError("运行清单缺少 Base 或数据表信息")
    table = tables.get(table_name)
    if not isinstance(table, dict):
        raise CaseMigrationError(f"运行清单缺少数据表: {table_name}")
    table_id = table.get("table_id")
    if not isinstance(table_id, str) or not table_id:
        raise CaseMigrationError(f"运行清单缺少 table_id: {table_name}")
    return base_id, table_id


def _options(field: dict[str, object]) -> list[dict[str, object]]:
    config = field.get("config")
    if not isinstance(config, dict):
        return []
    options = config.get("options")
    if not isinstance(options, list):
        return []
    return [dict(item) for item in options if isinstance(item, dict)]


def _records(payload: object) -> list[dict[str, object]]:
    value = payload.get("data", payload) if isinstance(payload, dict) else payload
    if isinstance(value, dict):
        value = value.get("records", value.get("items", []))
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _plain(value: object) -> str:
    if isinstance(value, dict):
        for name in ("markdown", "text", "value", "name"):
            candidate = value.get(name)
            if isinstance(candidate, str):
                return candidate
    if value is None:
        return ""
    return str(value)


def _field_spec(name: str) -> FieldSpec:
    for item in get_table_spec("知识缺口").fields:
        if item.name == name:
            return item
    raise CaseMigrationError(f"schema 缺少案件字段: {name}")


def build_case_migration_plan(
    client: Any,
    manifest: dict[str, object],
) -> list[CaseMigrationAction]:
    actions: list[CaseMigrationAction] = []
    base_id, gap_table_id = _table_info(manifest, "知识缺口")
    gap_fields = _table_fields(client.get_table(base_id, gap_table_id))
    existing_gap_names = {_field_name(item) for item in gap_fields}
    for field_name in CASE_GAP_FIELDS:
        if field_name not in existing_gap_names:
            actions.append(
                CaseMigrationAction(
                    "create_field",
                    "知识缺口",
                    field_name,
                    {"definition": _field_spec(field_name).as_dws_field()},
                )
            )

    for table_name in ("知识源台账", "知识条目"):
        _, table_id = _table_info(manifest, table_name)
        fields = _table_fields(client.get_table(base_id, table_id))
        domain = next(
            (item for item in fields if _field_name(item) == "一级业务域"),
            None,
        )
        if domain is None:
            raise CaseMigrationError(f"回读缺少一级业务域字段: {table_name}")
        field_id = _identifier(domain, "fieldId", "field_id", "id")
        if not field_id:
            raise CaseMigrationError(f"回读缺少一级业务域 fieldId: {table_name}")
        details = _table_fields(client.get_fields(base_id, table_id, (field_id,)))
        detailed_domain = next(
            (item for item in details if _identifier(item, "fieldId", "id") == field_id),
            None,
        )
        if detailed_domain is None:
            raise CaseMigrationError(f"回读缺少一级业务域完整配置: {table_name}")
        existing_options = _options(detailed_domain)
        options_by_name = {
            item.get("name"): item
            for item in existing_options
            if isinstance(item.get("name"), str)
        }
        if any(name not in options_by_name for name in BUSINESS_DOMAINS):
            actions.append(
                CaseMigrationAction(
                    "update_options",
                    table_name,
                    "一级业务域",
                    {
                        "field_id": field_id,
                        "config": {
                            "options": [
                                *[dict(item) for item in existing_options],
                                *[
                                    {"name": name}
                                    for name in BUSINESS_DOMAINS
                                    if name not in options_by_name
                                ],
                            ]
                        },
                    },
                )
            )

        table_definition = manifest["tables"][table_name]
        field_map = table_definition.get("fields") if isinstance(table_definition, dict) else None
        primary_name = "来源编号" if table_name == "知识源台账" else "知识编号"
        if not isinstance(field_map, dict):
            raise CaseMigrationError(f"运行清单字段无效: {table_name}")
        primary_id = field_map.get(primary_name)
        domain_id = field_map.get("一级业务域")
        if not isinstance(primary_id, str) or not isinstance(domain_id, str):
            raise CaseMigrationError(f"运行清单缺少业务编号或业务域: {table_name}")
        record_ids: list[str] = []
        for record in _records(client.query_all_records(base_id, table_id)):
            cells = record.get("cells")
            if not isinstance(cells, dict):
                continue
            business_id = _plain(cells.get(primary_id))
            current_domain = _plain(cells.get(domain_id))
            record_id = _identifier(record, "recordId", "record_id", "id")
            if business_id.startswith("EP3-") and not current_domain and record_id:
                record_ids.append(record_id)
        if record_ids:
            actions.append(
                CaseMigrationAction(
                    "backfill_domain",
                    table_name,
                    "一级业务域",
                    {
                        "record_ids": record_ids,
                        "field_id": domain_id,
                        "value": "员工合伙制",
                    },
                )
            )
    return actions


def apply_case_migration_plan(
    client: Any,
    manifest: dict[str, object],
    actions: list[CaseMigrationAction],
    manifest_path: Path,
) -> dict[str, object]:
    base_id = manifest.get("base_id")
    if not isinstance(base_id, str) or not base_id:
        raise CaseMigrationError("运行清单缺少 base_id")
    for action in actions:
        _, table_id = _table_info(manifest, action.table)
        if action.kind == "create_field":
            definition = action.payload.get("definition")
            if not isinstance(definition, dict):
                raise CaseMigrationError(f"字段迁移动作无效: {action.field}")
            client.create_fields(base_id, table_id, [definition])
        elif action.kind == "update_options":
            field_id = action.payload.get("field_id")
            config = action.payload.get("config")
            if not isinstance(field_id, str) or not isinstance(config, dict):
                raise CaseMigrationError(f"选项迁移动作无效: {action.table}")
            client.update_field(
                base_id,
                table_id,
                field_id,
                config=config,
            )
        elif action.kind == "backfill_domain":
            record_ids = action.payload.get("record_ids")
            field_id = action.payload.get("field_id")
            value = action.payload.get("value")
            if (
                not isinstance(record_ids, list)
                or not all(isinstance(item, str) for item in record_ids)
                or not isinstance(field_id, str)
                or not isinstance(value, str)
            ):
                raise CaseMigrationError(f"一级业务域回填动作无效: {action.table}")
            patches = [
                {"recordId": record_id, "cells": {field_id: value}}
                for record_id in record_ids
            ]
            with TemporaryDirectory(prefix="mumu-domain-recovery-") as directory:
                path = Path(directory) / "records.json"
                path.write_text(
                    json.dumps(patches, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                client.update_records(base_id, table_id, path)
        else:
            raise CaseMigrationError(f"未知迁移动作: {action.kind}")

    updated = deepcopy(manifest)
    tables = updated.get("tables")
    if not isinstance(tables, dict):
        raise CaseMigrationError("运行清单缺少数据表")
    for table_name in ("知识源台账", "知识条目", "知识缺口"):
        _, table_id = _table_info(updated, table_name)
        fields = _table_fields(client.get_table(base_id, table_id))
        field_map = {
            _field_name(item): _identifier(item, "fieldId", "field_id", "id")
            for item in fields
            if _field_name(item)
        }
        table_info = tables.get(table_name)
        if not isinstance(table_info, dict):
            raise CaseMigrationError(f"运行清单缺少数据表: {table_name}")
        table_info["fields"] = field_map

    gap_fields = tables["知识缺口"]["fields"]
    if not isinstance(gap_fields, dict):
        raise CaseMigrationError("知识缺口字段回读无效")
    missing = [name for name in CASE_GAP_FIELDS if not gap_fields.get(name)]
    if missing:
        raise CaseMigrationError(f"迁移回读缺少字段: {', '.join(missing)}")

    remaining = build_case_migration_plan(client, updated)
    if remaining:
        raise CaseMigrationError("迁移回读仍存在未完成动作")
    updated["schema_version"] = 3
    _write_manifest(updated, manifest_path)
    return updated
