from __future__ import annotations

import logging
import json
import re
import hashlib
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock, Thread
from typing import Any, Callable

from mumu_kb.models import (
    InterpretationBasis,
    InterpretationEntry,
    InterpretationStatus,
)
from mumu_kb.payloads import build_interpretation_cells

from .domain import Evidence
from .repository import _cell_text, _date, _lines, _questions


logger = logging.getLogger(__name__)


def _records(payload: object) -> list[dict[str, object]]:
    value = payload.get("data", payload) if isinstance(payload, dict) else payload
    if isinstance(value, dict):
        value = value.get("records", value.get("items", []))
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _identifier(record: dict[str, object]) -> str:
    for name in ("recordId", "record_id", "id"):
        value = record.get(name)
        if isinstance(value, str) and value:
            return value
    return ""


def _linked_knowledge_ids(value: object) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            item.strip()
            for item in re.split(r"[\s,，;；]+", _cell_text(value))
            if item.strip()
        )
    )


class DwsInterpretationStore:
    def __init__(self, client: Any, manifest: dict[str, object]) -> None:
        self.client = client
        self.manifest = manifest
        base_id = manifest.get("base_id")
        tables = manifest.get("tables")
        if not isinstance(base_id, str) or not isinstance(tables, dict):
            raise ValueError("运行清单缺少 Base 或数据表")
        self.base_id = base_id
        self.tables: dict[str, tuple[str, dict[str, str]]] = {}
        for name in ("知识条目", "制度解读"):
            table = tables.get(name)
            if not isinstance(table, dict):
                raise ValueError(f"运行清单缺少数据表: {name}")
            table_id = table.get("table_id")
            fields = table.get("fields")
            if not isinstance(table_id, str) or not isinstance(fields, dict):
                raise ValueError(f"运行清单数据表无效: {name}")
            self.tables[name] = (
                table_id,
                {
                    str(field_name): str(field_id)
                    for field_name, field_id in fields.items()
                    if isinstance(field_name, str)
                    and isinstance(field_id, str)
                    and field_id
                },
            )

    def _all(self, table_name: str) -> list[dict[str, object]]:
        table_id, _ = self.tables[table_name]
        return _records(self.client.query_all_records(self.base_id, table_id))

    def _cell(
        self,
        table_name: str,
        record: dict[str, object],
        field_name: str,
    ) -> object:
        _, fields = self.tables[table_name]
        cells = record.get("cells")
        if not isinstance(cells, dict):
            return None
        field_id = fields.get(field_name)
        return cells.get(field_id) if field_id else None

    def list_pending(self, now: datetime, *, limit: int) -> tuple[Evidence, ...]:
        if limit < 1:
            return ()
        covered: set[tuple[str, str]] = set()
        for record in self._all("制度解读"):
            version = _cell_text(self._cell("制度解读", record, "关联制度版本")) or "1.0"
            for knowledge_id in _linked_knowledge_ids(
                self._cell("制度解读", record, "关联知识编号")
            ):
                covered.add((knowledge_id, version))

        pending: list[Evidence] = []
        for record in self._all("知识条目"):
            if _cell_text(self._cell("知识条目", record, "发布状态")) != "已发布":
                continue
            if not _cell_text(self._cell("知识条目", record, "入库确认人")):
                continue
            try:
                effective_from = _date(self._cell("知识条目", record, "生效时间"))
                effective_to = _date(self._cell("知识条目", record, "失效时间"))
            except ValueError:
                continue
            if effective_from is not None and now < effective_from:
                continue
            if effective_to is not None and now > effective_to:
                continue
            knowledge_id = _cell_text(self._cell("知识条目", record, "知识编号"))
            version = _cell_text(self._cell("知识条目", record, "适用制度版本")) or "1.0"
            if not knowledge_id or (knowledge_id, version) in covered:
                continue
            evidence = Evidence(
                knowledge_id=knowledge_id,
                intent=_cell_text(self._cell("知识条目", record, "用户意图")),
                answer=_cell_text(self._cell("知识条目", record, "标准答案")),
                source_id=_cell_text(self._cell("知识条目", record, "来源编号")),
                source_section=_cell_text(self._cell("知识条目", record, "来源章节")),
                version=version,
                risk_level=_cell_text(self._cell("知识条目", record, "风险等级")),
                sensitivity=_cell_text(self._cell("知识条目", record, "信息可见范围")),
                conditions=_lines(self._cell("知识条目", record, "成立条件")),
                exceptions=_lines(self._cell("知识条目", record, "例外情形")),
                steps=_lines(self._cell("知识条目", record, "办理步骤")),
                common_questions=_questions(
                    self._cell("知识条目", record, "常见问法")
                ),
                business_domain=(
                    _cell_text(self._cell("知识条目", record, "一级业务域"))
                    or "公司通用"
                ),
            )
            if not evidence.answer or not evidence.source_id or not evidence.source_section:
                continue
            pending.append(evidence)
            if len(pending) >= limit:
                break
        return tuple(pending)

    def save(self, entry: InterpretationEntry) -> None:
        entry.validate()
        table_id, fields = self.tables["制度解读"]
        existing = next(
            (
                record
                for record in self._all("制度解读")
                if _cell_text(
                    (record.get("cells") or {}).get(fields["解读编号"])
                    if isinstance(record.get("cells"), dict)
                    else None
                )
                == entry.interpretation_id
            ),
            None,
        )
        payload: dict[str, object] = {
            "cells": build_interpretation_cells(entry, self.manifest)
        }
        update = existing is not None
        if existing is not None:
            record_id = _identifier(existing)
            if not record_id:
                raise ValueError("已有制度解读缺少 recordId")
            payload["recordId"] = record_id
        with TemporaryDirectory(prefix="mumu-interpretation-enrichment-") as directory:
            path = Path(directory) / "records.json"
            path.write_text(
                json.dumps([payload], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if update:
                self.client.update_records(self.base_id, table_id, path)
            else:
                self.client.create_records(self.base_id, table_id, path)

    def mark_not_needed(self, evidence: Evidence) -> None:
        digest = hashlib.sha256(
            f"skip|{evidence.knowledge_id}|{evidence.version}".encode("utf-8")
        ).hexdigest()[:12].upper()
        self.save(
            InterpretationEntry(
                interpretation_id=f"MUMU-I-{digest}",
                knowledge_ids=[evidence.knowledge_id],
                business_domain=evidence.business_domain or "公司通用",
                audience="员工",
                employee_concern="该知识是否需要额外解释？",
                rationale="该条目属于直接事实问答，无需额外制度解读。",
                employee_value="直接提供正式答案更清晰。",
                employee_cost="无额外解读。",
                recommended_response="按正式知识直接回答。",
                basis=InterpretationBasis.DERIVED,
                source_label="木木自动判断：无需解读",
                source_section=evidence.source_section,
                status=InterpretationStatus.DISABLED,
                boundaries=["不得脱离正式知识扩展解释"],
                version=evidence.version or "1.0",
            )
        )


class InterpretationEnrichmentWorker:
    def __init__(
        self,
        *,
        store: Any,
        model: Any,
        knowledge_repository: Any | None = None,
        batch_size: int = 20,
    ) -> None:
        if batch_size < 1:
            raise ValueError("解读补全批次必须为正整数")
        self.store = store
        self.model = model
        self.knowledge_repository = knowledge_repository
        self.batch_size = batch_size
        self._run_lock = Lock()

    def run_once(self, now: datetime) -> int:
        if not self._run_lock.acquire(blocking=False):
            return 0
        completed = 0
        published = 0
        try:
            for evidence in self.store.list_pending(now, limit=self.batch_size):
                try:
                    entry = self.model.generate_interpretation(evidence)
                    if entry is None:
                        self.store.mark_not_needed(evidence)
                        completed += 1
                        continue
                    entry.validate()
                    if entry.knowledge_ids != [evidence.knowledge_id]:
                        raise ValueError("自动解读关联了错误的知识编号")
                    if entry.version != evidence.version:
                        raise ValueError("自动解读版本与正式知识不一致")
                    self.store.save(entry)
                    completed += 1
                    published += 1
                except Exception as error:
                    logger.warning(
                        "interpretation_enrichment_failed knowledge_id=%s error_type=%s",
                        evidence.knowledge_id,
                        type(error).__name__,
                    )
            if published and self.knowledge_repository is not None:
                self.knowledge_repository.invalidate_snapshot()
            return completed
        finally:
            self._run_lock.release()


class InterpretationEnrichmentRunner:
    def __init__(
        self,
        worker: InterpretationEnrichmentWorker,
        *,
        interval_seconds: float = 30,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("制度解读扫描间隔必须为正数")
        self.worker = worker
        self.interval_seconds = interval_seconds
        self.clock = clock or (lambda: datetime.now().astimezone())
        self.stop_event = Event()
        self.thread: Thread | None = None

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = Thread(
            target=self._run,
            name="mumu-interpretation-enrichment",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(1.0, self.interval_seconds + 1))

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.worker.run_once(self.clock())
            except Exception as error:
                logger.warning(
                    "interpretation_enrichment_scan_failed error_type=%s",
                    type(error).__name__,
                )
            self.stop_event.wait(self.interval_seconds)
