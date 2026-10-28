from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .provision import _write_manifest
from .schema import get_table_spec


class InterpretationMigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class InterpretationMigrationAction:
    kind: str
    payload: dict[str, object]


def _items(payload: object, *keys: str) -> list[dict[str, object]]:
    value = payload.get("data", payload) if isinstance(payload, dict) else payload
    if isinstance(value, dict):
        for key in keys:
            candidate = value.get(key)
            if isinstance(candidate, list):
                return [item for item in candidate if isinstance(item, dict)]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _identifier(item: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _name(item: dict[str, object]) -> str:
    return _identifier(item, "name", "tableName", "fieldName")


def _table_rows(payload: object) -> list[dict[str, object]]:
    return _items(payload, "tables", "items", "records")


def _field_rows(payload: object) -> list[dict[str, object]]:
    tables = _table_rows(payload)
    if tables:
        fields = tables[0].get("fields")
        if isinstance(fields, list):
            return [item for item in fields if isinstance(item, dict)]
    return _items(payload, "fields", "items", "records")


def _find_table(client: Any, base_id: str) -> tuple[str, list[dict[str, object]]]:
    for table in _table_rows(client.get_base(base_id)):
        if _name(table) != "制度解读":
            continue
        table_id = _identifier(table, "tableId", "table_id", "id")
        if not table_id:
            raise InterpretationMigrationError("制度解读表缺少 tableId")
        return table_id, _field_rows(client.get_table(base_id, table_id))
    return "", []


def build_interpretation_migration_plan(
    client: Any,
    manifest: dict[str, object],
) -> list[InterpretationMigrationAction]:
    base_id = manifest.get("base_id")
    if not isinstance(base_id, str) or not base_id:
        raise InterpretationMigrationError("运行清单缺少 base_id")
    table_id, fields = _find_table(client, base_id)
    spec = get_table_spec("制度解读")
    if not table_id:
        return [
            InterpretationMigrationAction(
                "create_table",
                {"fields": [field.as_dws_field() for field in spec.fields]},
            )
        ]
    existing = {_name(field) for field in fields}
    missing = [field for field in spec.fields if field.name not in existing]
    return [
        InterpretationMigrationAction(
            "create_fields",
            {
                "table_id": table_id,
                "fields": [field.as_dws_field() for field in missing],
            },
        )
    ] if missing else []


def apply_interpretation_migration_plan(
    client: Any,
    manifest: dict[str, object],
    actions: list[InterpretationMigrationAction],
    manifest_path: Path,
) -> dict[str, object]:
    base_id = manifest.get("base_id")
    if not isinstance(base_id, str) or not base_id:
        raise InterpretationMigrationError("运行清单缺少 base_id")
    for action in actions:
        if action.kind == "create_table":
            fields = action.payload.get("fields")
            if not isinstance(fields, list):
                raise InterpretationMigrationError("建表字段无效")
            client.create_table(base_id, "制度解读", fields)
        elif action.kind == "create_fields":
            table_id = action.payload.get("table_id")
            fields = action.payload.get("fields")
            if not isinstance(table_id, str) or not isinstance(fields, list):
                raise InterpretationMigrationError("补充字段动作无效")
            for start in range(0, len(fields), 15):
                client.create_fields(base_id, table_id, fields[start : start + 15])
        else:
            raise InterpretationMigrationError(f"未知迁移动作: {action.kind}")

    table_id, field_rows = _find_table(client, base_id)
    if not table_id:
        raise InterpretationMigrationError("迁移后未找到制度解读表")
    field_map = {
        _name(field): _identifier(field, "fieldId", "field_id", "id")
        for field in field_rows
    }
    spec = get_table_spec("制度解读")
    missing = [field.name for field in spec.fields if not field_map.get(field.name)]
    if missing:
        raise InterpretationMigrationError(
            "制度解读表回读缺少字段: " + "、".join(missing)
        )
    updated = deepcopy(manifest)
    tables = updated.setdefault("tables", {})
    if not isinstance(tables, dict):
        raise InterpretationMigrationError("运行清单 tables 格式无效")
    tables["制度解读"] = {
        "table_id": table_id,
        "fields": {field.name: field_map[field.name] for field in spec.fields},
    }
    updated["schema_version"] = 4
    _write_manifest(updated, manifest_path)
    return updated
