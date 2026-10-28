from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import TABLE_SPECS, FieldSpec, TableSpec


class ProvisionConflict(RuntimeError):
    pass


class ProvisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProvisionAction:
    kind: str
    table: str = ""
    field: str = ""
    payload: dict[str, object] | None = None


def _data(payload: object) -> object:
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def _items(payload: object, names: tuple[str, ...]) -> list[dict[str, object]]:
    value = _data(payload)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for name in names:
            candidate = value.get(name)
            if isinstance(candidate, list):
                return [item for item in candidate if isinstance(item, dict)]
    return []


def _value(payload: object, names: tuple[str, ...]) -> str:
    value = _data(payload)
    if isinstance(value, dict):
        for name in names:
            candidate = value.get(name)
            if isinstance(candidate, str) and candidate:
                return candidate
    return ""


def _name(item: dict[str, object]) -> str:
    for key in ("baseName", "tableName", "fieldName", "name"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return ""


def _identifier(item: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def build_provision_plan(
    client: Any,
    base_name: str = "木木知识库",
    reuse_base_id: str | None = None,
) -> list[ProvisionAction]:
    search_result = client.search_bases(base_name)
    candidates = [
        item
        for item in _items(search_result, ("items", "bases", "records"))
        if _name(item) == base_name
    ]
    if candidates and reuse_base_id is None:
        ids = [
            _identifier(item, "baseId", "base_id", "id") or "未知ID"
            for item in candidates
        ]
        raise ProvisionConflict(
            f"已存在同名 Base {base_name}: {', '.join(ids)}；必须显式指定复用 ID"
        )
    if reuse_base_id is not None:
        if not any(
            _identifier(item, "baseId", "base_id", "id") == reuse_base_id
            for item in candidates
        ):
            raise ProvisionConflict(f"指定复用的 Base 不在同名搜索结果中: {reuse_base_id}")
        return [
            ProvisionAction(
                "reuse_base",
                payload={"base_id": reuse_base_id, "base_name": base_name},
            )
        ]

    actions = [
        ProvisionAction("create_base", payload={"name": base_name}),
        ProvisionAction("rename_default_table", table="知识源台账"),
        ProvisionAction(
            "configure_source_fields",
            table="知识源台账",
            payload={"fields": list(TABLE_SPECS[0].fields)},
        ),
    ]
    actions.extend(
        ProvisionAction(
            "create_table",
            table=table.name,
            payload={"fields": list(table.fields)},
        )
        for table in TABLE_SPECS[1:]
    )
    return actions


def _field_payloads(fields: list[FieldSpec]) -> list[dict[str, object]]:
    return [field.as_dws_field() for field in fields]


def _table_rows(payload: object) -> list[dict[str, object]]:
    return _items(payload, ("tables", "items", "records"))


def _field_rows(payload: object) -> list[dict[str, object]]:
    tables = _table_rows(payload)
    if tables:
        fields = tables[0].get("fields")
        if isinstance(fields, list):
            return [item for item in fields if isinstance(item, dict)]
    return _items(payload, ("fields", "items", "records"))


def _find_table_id(client: Any, base_id: str, table_name: str) -> str:
    for table in _table_rows(client.get_base(base_id)):
        if _name(table) == table_name:
            table_id = _identifier(table, "tableId", "table_id", "id")
            if table_id:
                return table_id
    raise ProvisionError(f"无法从 Base 回读数据表 ID: {table_name}")


def _load_table_fields(client: Any, base_id: str, table_id: str) -> list[dict[str, object]]:
    fields = _field_rows(client.get_table(base_id, table_id))
    if not fields:
        raise ProvisionError(f"无法回读字段: {table_id}")
    return fields


def _create_field_batches(
    client: Any,
    base_id: str,
    table_id: str,
    fields: list[FieldSpec],
) -> None:
    for start in range(0, len(fields), 15):
        client.create_fields(
            base_id,
            table_id,
            _field_payloads(fields[start : start + 15]),
        )


def _build_manifest(client: Any, base_id: str, base_name: str) -> dict[str, object]:
    manifest_tables: dict[str, object] = {}
    for table_spec in TABLE_SPECS:
        table_id = _find_table_id(client, base_id, table_spec.name)
        field_rows = _load_table_fields(client, base_id, table_id)
        field_map = {
            _name(field): _identifier(field, "fieldId", "field_id", "id")
            for field in field_rows
        }
        expected_names = {field.name for field in table_spec.fields}
        missing = sorted(name for name in expected_names if not field_map.get(name))
        if missing:
            raise ProvisionError(
                f"数据表 {table_spec.name} 回读缺少字段: {', '.join(missing)}"
            )
        manifest_tables[table_spec.name] = {
            "table_id": table_id,
            "fields": {name: field_map[name] for name in expected_names},
        }
    return {
        "schema_version": 5,
        "base_name": base_name,
        "base_id": base_id,
        "tables": manifest_tables,
    }


def _write_manifest(manifest: dict[str, object], manifest_path: Path) -> None:
    resolved = manifest_path.resolve()
    if ".runtime" not in resolved.parts or resolved.name != "aitable-manifest.json":
        raise ProvisionError("运行清单必须写入 .runtime/aitable-manifest.json")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)


def apply_provision_plan(
    client: Any,
    actions: list[ProvisionAction],
    manifest_path: Path,
) -> dict[str, object]:
    base_id = ""
    base_name = "木木知识库"
    source_table_id = ""
    for action in actions:
        payload = action.payload or {}
        if action.kind == "reuse_base":
            base_id = str(payload["base_id"])
            base_name = str(payload["base_name"])
            continue
        if action.kind == "create_base":
            base_name = str(payload["name"])
            response = client.create_base(base_name)
            base_id = _value(response, ("baseId", "base_id", "id"))
            if not base_id:
                raise ProvisionError("创建 Base 后未返回 baseId")
            base_tables = _table_rows(client.get_base(base_id))
            if len(base_tables) != 1:
                raise ProvisionError("新 Base 必须且只能包含一张默认表")
            source_table_id = _identifier(base_tables[0], "tableId", "table_id", "id")
        elif action.kind == "rename_default_table":
            if not base_id or not source_table_id:
                raise ProvisionError("默认表重命名动作缺少 Base 或 Table ID")
            client.update_table(base_id, source_table_id, action.table)
        elif action.kind == "configure_source_fields":
            fields = list(payload["fields"])
            existing = _load_table_fields(client, base_id, source_table_id)
            primary_id = _identifier(existing[0], "fieldId", "field_id", "id")
            client.update_field(
                base_id,
                source_table_id,
                primary_id,
                name=fields[0].name,
            )
            _create_field_batches(client, base_id, source_table_id, fields[1:])
        elif action.kind == "create_table":
            fields = list(payload["fields"])
            response = client.create_table(
                base_id,
                action.table,
                _field_payloads(fields[:15]),
            )
            table_id = _value(response, ("tableId", "table_id", "id"))
            if not table_id:
                table_id = _find_table_id(client, base_id, action.table)
            _create_field_batches(client, base_id, table_id, fields[15:])
        else:
            raise ProvisionError(f"未知 Provision 动作: {action.kind}")

    if not base_id:
        raise ProvisionError("Provision 计划没有解析出 Base ID")
    manifest = _build_manifest(client, base_id, base_name)
    _write_manifest(manifest, manifest_path)
    return manifest


def load_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvisionError(f"无法读取运行清单: {path}") from error
    if not isinstance(value, dict):
        raise ProvisionError("运行清单顶层必须是对象")
    return value
