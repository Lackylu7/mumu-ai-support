from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .models import InterpretationEntry, KnowledgeEntry, load_json_records
from .payloads import build_interpretation_cells


class InterpretationImportError(RuntimeError):
    pass


@dataclass(frozen=True)
class InterpretationSyncSummary:
    created: int
    updated: int
    remote_count: int


def _records(payload: object) -> list[dict[str, object]]:
    value = payload.get("data", payload) if isinstance(payload, dict) else payload
    if isinstance(value, dict):
        value = value.get("records", value.get("items", []))
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _identifier(item: dict[str, object]) -> str:
    for key in ("recordId", "record_id", "id"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _plain(value: object) -> str:
    if isinstance(value, dict):
        for key in ("markdown", "text", "value", "name"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate.strip()
    return value.strip() if isinstance(value, str) else ""


def load_interpretation_pack(
    interpretation_path: Path,
    knowledge_path: Path,
) -> list[InterpretationEntry]:
    entries = load_json_records(interpretation_path, InterpretationEntry)
    knowledge = load_json_records(knowledge_path, KnowledgeEntry)
    knowledge_ids = {entry.knowledge_id for entry in knowledge}
    for entry in entries:
        missing = sorted(set(entry.knowledge_ids) - knowledge_ids)
        if missing:
            raise InterpretationImportError(
                f"{entry.interpretation_id} 引用不存在知识: {', '.join(missing)}"
            )
    return entries


def sync_interpretations(
    client: Any,
    manifest: dict[str, object],
    entries: list[InterpretationEntry],
    *,
    confirm: bool,
) -> InterpretationSyncSummary:
    if not confirm:
        raise InterpretationImportError("写入制度解读需要显式确认")
    try:
        base_id = str(manifest["base_id"])
        table = manifest["tables"]["制度解读"]
        table_id = str(table["table_id"])
        primary_field_id = str(table["fields"]["解读编号"])
    except (KeyError, TypeError) as error:
        raise InterpretationImportError("运行清单缺少制度解读表") from error

    remote = _records(client.query_all_records(base_id, table_id))
    by_interpretation_id: dict[str, str] = {}
    for record in remote:
        cells = record.get("cells")
        if not isinstance(cells, dict):
            continue
        interpretation_id = _plain(cells.get(primary_field_id))
        record_id = _identifier(record)
        if interpretation_id and record_id:
            by_interpretation_id[interpretation_id] = record_id

    creates: list[dict[str, object]] = []
    updates: list[dict[str, object]] = []
    for entry in entries:
        cells = build_interpretation_cells(entry, manifest)
        record_id = by_interpretation_id.get(entry.interpretation_id)
        if record_id:
            updates.append({"recordId": record_id, "cells": cells})
        else:
            creates.append({"cells": cells})

    with TemporaryDirectory(prefix="mumu-interpretation-sync-") as directory:
        root = Path(directory)
        for operation, records in (("create", creates), ("update", updates)):
            for start in range(0, len(records), 30):
                batch = records[start : start + 30]
                path = root / f"{operation}-{start // 30 + 1:03d}.json"
                path.write_text(
                    json.dumps(batch, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                if operation == "create":
                    client.create_records(base_id, table_id, path)
                else:
                    client.update_records(base_id, table_id, path)

    after = _records(client.query_all_records(base_id, table_id))
    return InterpretationSyncSummary(
        created=len(creates),
        updated=len(updates),
        remote_count=len(after),
    )
