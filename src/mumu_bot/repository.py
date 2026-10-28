from __future__ import annotations

import re
import logging
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Callable, Protocol

from .deepseek import DomainRoutingDecision, EvidenceSelection
from .domain import (
    ConversationTurn,
    Evidence,
    InterpretationEvidence,
    QuestionScope,
    RetrievalResult,
)
from .v2_intent import (
    v2_domain_values,
    v2_keyword_variants,
    v2_topic_values,
)


class RecordClient(Protocol):
    def query_all_records(
        self,
        base_id: str,
        table_id: str,
    ) -> dict[str, object]: ...

    def query_records(
        self,
        *,
        base_id: str,
        table_id: str,
        field_ids: list[str],
        filters: dict[str, object],
        query: str,
        limit: int,
    ) -> dict[str, object]: ...


class StoreGuideClient(Protocol):
    def search_wiki_nodes(
        self,
        workspace_id: str,
        query: str,
        *,
        limit: int = 8,
    ) -> dict[str, object]: ...

    def read_doc(self, node_id: str) -> dict[str, object]: ...


class KnowledgeRetriever(Protocol):
    def retrieve(
        self,
        question: str,
        now: datetime,
        context: Sequence[ConversationTurn] = (),
        source_mode: str = "both",
        live_sources: Sequence[str] = (),
        live_search_policy: str = "single",
    ) -> RetrievalResult: ...


class EvidenceSelector(Protocol):
    def route_domains(
        self,
        question: str,
        domain_profiles: dict[str, tuple[str, ...]],
    ) -> DomainRoutingDecision: ...

    def select_evidence(
        self,
        question: str,
        candidates: list[Evidence],
        limit: int = 5,
    ) -> tuple[str, ...]: ...

    def analyze_evidence(
        self,
        question: str,
        candidates: list[Evidence],
        limit: int = 5,
    ) -> EvidenceSelection: ...


logger = logging.getLogger(__name__)

INITIAL_SNAPSHOT_ATTEMPTS = 3
INITIAL_SNAPSHOT_RETRY_SECONDS = 0.5
MAX_PARALLEL_SELECTOR_WORKERS = 4
SELECTION_CACHE_CAPACITY = 256


def _cell_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        markdown = value.get("markdown")
        if isinstance(markdown, str):
            return markdown.strip()
        name = value.get("name")
        if isinstance(name, str):
            return name.strip()
    return ""


def _lines(value: object) -> tuple[str, ...]:
    return tuple(line.strip() for line in _cell_text(value).splitlines() if line.strip())


def _date(value: object) -> datetime | None:
    text = _cell_text(value)
    if not text:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


_QUERY_REPLACEMENTS = (
    ("多少钱", "金额"),
    ("多少", "金额"),
    ("每家门店", "单店"),
    ("每个门店", "单店"),
    ("每家店", "单店"),
    ("什么时候", "时间"),
    ("何时", "时间"),
    ("哪些事项", "用途"),
    ("用于哪些", "用途"),
    ("用于什么", "用途"),
    ("用在哪里", "用途"),
    ("怎么处理", "处理"),
    ("怎么办", "处理"),
    ("怎样", "如何"),
    ("能不能", "能否"),
    ("可不可以", "能否"),
    ("有没有", "是否有"),
    ("是不是", "是否"),
)

_QUERY_STOP_WORDS = (
    "请告诉我",
    "请说明",
    "请问",
    "到底",
    "具体",
    "现在",
    "相关",
    "问题",
    "一下",
    "什么",
    "哪些",
    "如何",
    "是否",
    "吗",
    "呢",
    "的",
    "了",
)


def _normalized(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    for source, target in _QUERY_REPLACEMENTS:
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"[^0-9a-z%\u4e00-\u9fff]+", "", normalized)
    for word in _QUERY_STOP_WORDS:
        normalized = normalized.replace(word, "")
    return normalized


def _exact_question_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    return re.sub(r"[^0-9a-z%\u4e00-\u9fff]+", "", normalized)


def _question_mapping_candidates(
    question: str,
    candidates: Sequence[Evidence],
) -> list[Evidence]:
    question_key = _exact_question_key(question)
    exact_mappings = [
        item
        for item in candidates
        if question_key
        and any(
            _exact_question_key(common_question) == question_key
            for common_question in item.common_questions
        )
    ]
    if exact_mappings:
        return exact_mappings

    # Only drop harmless interrogative fillers. A unique contained match is more
    # precise than a broad semantic merge for questions such as “OA 应该由谁提交”.
    def key_without_fillers(value: str) -> str:
        key = _exact_question_key(value)
        for filler in ("请问", "麻烦", "一下", "应该", "需要", "由", "要"):
            key = key.replace(filler, "")
        return key

    normalized_question = key_without_fillers(question)
    contained_mappings = [
        item
        for item in candidates
        if any(
            (common_key := key_without_fillers(common_question))
            and len(common_key) >= 6
            and common_key in normalized_question
            for common_question in item.common_questions
        )
    ]
    return contained_mappings if len(contained_mappings) == 1 else []


def _is_business_overview_request(question: str) -> bool:
    compact = re.sub(r"\s+", "", question)
    return bool(
        re.search(
            r"(?:讲讲|介绍|概览|全流程|完整流程|整体|都有哪些|哪些方面)",
            compact,
        )
    )


_GENERIC_DOMAIN_TOKENS = frozenset({"公司", "通用", "业务", "运营", "管理", "制度"})


def _explicitly_named_domains(
    question: str,
    domain_profiles: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    normalized_question = _normalized(question)
    if not normalized_question:
        return ()
    matched: list[str] = []
    for domain in domain_profiles:
        if domain == "公司通用":
            continue
        normalized_domain = _normalized(domain)
        if not normalized_domain:
            continue
        domain_terms = (
            {normalized_domain}
            | {
                normalized_domain[index : index + 2]
                for index in range(len(normalized_domain) - 1)
            }
        ) - _GENERIC_DOMAIN_TOKENS
        if any(len(term) >= 2 and term in normalized_question for term in domain_terms):
            matched.append(domain)
    return tuple(matched)


def _bigrams(value: str) -> set[str]:
    return {value[index : index + 2] for index in range(len(value) - 1)}


def _similarity(left: str, right: str) -> float:
    normalized_left = _normalized(left)
    normalized_right = _normalized(right)
    if not normalized_left or not normalized_right:
        return 0.0
    sequence_score = SequenceMatcher(
        None,
        normalized_left,
        normalized_right,
    ).ratio()
    left_bigrams = _bigrams(normalized_left)
    right_bigrams = _bigrams(normalized_right)
    bigram_score = 0.0
    if left_bigrams and right_bigrams:
        bigram_score = (
            2 * len(left_bigrams & right_bigrams)
            / (len(left_bigrams) + len(right_bigrams))
        )
    containment_score = 0.0
    if normalized_left in normalized_right or normalized_right in normalized_left:
        containment_score = (
            min(len(normalized_left), len(normalized_right))
            / max(len(normalized_left), len(normalized_right))
        )
    return max(sequence_score, bigram_score, containment_score)


def _questions(value: object) -> tuple[str, ...]:
    text = _cell_text(value)
    return tuple(
        part.strip()
        for part in re.split(r"(?<=[？！。!?])\s*|\n+", text)
        if part.strip()
    )


class DwsKnowledgeRepository:
    def __init__(
        self,
        client: RecordClient,
        *,
        base_id: str,
        table_id: str,
        fields: Mapping[str, str],
        limit: int = 8,
        selector: EvidenceSelector | None = None,
        retrieval_mode: str = "legacy",
        snapshot_ttl_seconds: int = 300,
        selector_batch_size: int = 40,
        domain_routing_enabled: bool = False,
        domain_routing_min_confidence: float = 0.85,
        interpretation_table_id: str = "",
        interpretation_fields: Mapping[str, str] | None = None,
        reference_table_id: str = "",
        reference_fields: Mapping[str, str] | None = None,
        source_table_id: str = "",
        source_fields: Mapping[str, str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if retrieval_mode not in {"legacy", "full_table", "v2_live"}:
            raise ValueError("retrieval_mode 必须为 legacy、full_table 或 v2_live")
        if snapshot_ttl_seconds < 1 or selector_batch_size < 1:
            raise ValueError("快照刷新时间和 AI 批次大小必须为正整数")
        if not 0.0 <= domain_routing_min_confidence <= 1.0:
            raise ValueError("业务分流最低置信度必须在 0 到 1 之间")
        self.client = client
        self.base_id = base_id
        self.table_id = table_id
        self.fields = dict(fields)
        self.limit = limit
        self.selector = selector
        self.retrieval_mode = retrieval_mode
        self.snapshot_ttl_seconds = snapshot_ttl_seconds
        self.selector_batch_size = selector_batch_size
        self.domain_routing_enabled = domain_routing_enabled
        self.domain_routing_min_confidence = domain_routing_min_confidence
        self.interpretation_table_id = interpretation_table_id
        self.interpretation_fields = dict(interpretation_fields or {})
        self.reference_table_id = reference_table_id
        self.reference_fields = dict(reference_fields or {})
        self.source_table_id = source_table_id
        self.source_fields = dict(source_fields or {})
        self.clock = clock
        self._snapshot_records: list[object] | None = None
        self._snapshot_loaded_at = 0.0
        self._snapshot_lock = Lock()
        self._interpretation_snapshot_records: list[object] | None = None
        self._interpretation_snapshot_loaded_at = 0.0
        self._interpretation_snapshot_lock = Lock()
        self._selection_cache: OrderedDict[tuple[str, float], tuple[str, ...]] = OrderedDict()
        self._selection_cache_lock = Lock()
        self._source_links: dict[str, str] | None = None
        self._source_links_loaded_at = 0.0
        self._source_links_lock = Lock()

    def retrieve(
        self,
        question: str,
        now: datetime,
        context: Sequence[ConversationTurn] = (),
        source_mode: str = "both",
        live_sources: Sequence[str] = (),
    ) -> RetrievalResult:
        del source_mode, live_sources
        if self.retrieval_mode == "v2_live":
            result = self._retrieve_v2_live(question, now, context)
            return self._with_reference_fallback(question, now, result)
        if self.retrieval_mode != "full_table":
            evidence = self._search_legacy(question, now)
            if evidence:
                return RetrievalResult(
                    QuestionScope.BUSINESS_MATCHED,
                    tuple(self._attach_interpretations(evidence)),
                )
            return RetrievalResult(
                QuestionScope.BUSINESS_GAP,
                reason="公司业务知识不足",
            )
        return self._retrieve_full_table(question, now, context)

    def _with_reference_fallback(
        self,
        question: str,
        now: datetime,
        result: RetrievalResult,
    ) -> RetrievalResult:
        """Use the broader reference library only when formal knowledge is absent."""
        if result.scope is not QuestionScope.BUSINESS_GAP:
            return result
        if not self.reference_table_id or not self.reference_fields or self.selector is None:
            return result
        try:
            payload = self.client.query_all_records(self.base_id, self.reference_table_id)
            candidates = self._parse_reference_records(self._records(payload))
            if not candidates:
                return result
            selected_ids = self.selector.select_evidence(question, candidates, limit=3)
            by_id = {item.knowledge_id: item for item in candidates}
            selected = [by_id[item_id] for item_id in selected_ids if item_id in by_id]
            if not selected:
                return result
            return RetrievalResult(QuestionScope.BUSINESS_MATCHED, tuple(selected))
        except Exception as error:
            logger.warning("reference_library_fallback_failed error_type=%s", type(error).__name__)
            return result

    def _parse_reference_records(self, records: list[object]) -> list[Evidence]:
        def field(cells: Mapping[str, object], name: str) -> object:
            field_id = self.reference_fields.get(name, "")
            return cells.get(field_id) if field_id else None

        evidence: list[Evidence] = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            cells = record.get("cells")
            if not isinstance(cells, Mapping):
                continue
            if _cell_text(field(cells, "接入状态")) != "已接入":
                continue
            reference_id = _cell_text(field(cells, "参考编号"))
            content = _cell_text(field(cells, "原文内容"))
            if not reference_id or not content:
                continue
            title = _cell_text(field(cells, "资料标题"))
            boundary = _cell_text(field(cells, "回答边界")) or "需确认当前规则"
            evidence.append(Evidence(
                knowledge_id=reference_id,
                intent=title or reference_id,
                answer=content,
                source_id=title or "资料参考库",
                source_section=_cell_text(field(cells, "来源章节")),
                version="资料参考库 2026-08",
                risk_level="普通",
                sensitivity="公开内部",
                business_domain="公司通用",
                business_topic=_cell_text(field(cells, "资料类型")) or "参考资料",
                answer_boundary=boundary,
            ))
        return evidence

    def search(self, question: str, now: datetime) -> list[Evidence]:
        if self.retrieval_mode == "v2_live":
            result = self._retrieve_v2_live(question, now)
            return list(result.evidence)
        if self.retrieval_mode == "full_table":
            return self._search_full_table(question, now)

        return self._attach_interpretations(self._search_legacy(question, now))

    def _retrieve_v2_live(
        self,
        question: str,
        now: datetime,
        context: Sequence[ConversationTurn] = (),
    ) -> RetrievalResult:
        """Run the v2 retrieval ladder against the current AI-table contents.

        This deliberately has no snapshot or selection cache.  A knowledge edit is
        therefore visible to the very next question instead of waiting for a TTL.
        """
        started_at = time.perf_counter()
        semantic_question = self._contextual_question(question, context)
        analyze = getattr(self.selector, "analyze_evidence", None)
        if not callable(analyze):
            return RetrievalResult(
                QuestionScope.TEMPORARY_ERROR,
                reason="语义检索暂时不可用",
            )

        scopes: list[QuestionScope] = []
        stages: list[tuple[str, list[Evidence]]] = []
        try:
            for topic in v2_topic_values(question):
                stages.append(
                    (
                        "topic",
                        self._query_live_records(
                            now,
                            query="",
                            field_name="二级主题",
                            field_value=topic,
                        ),
                    )
                )
            stages.append(("question", self._query_live_records(now, query=question)))
            for keyword in v2_keyword_variants(question):
                if keyword != question:
                    stages.append(("keyword", self._query_live_records(now, query=keyword)))
            for domain in v2_domain_values(question):
                stages.append(
                    (
                        "domain",
                        self._query_live_records(
                            now,
                            query="",
                            field_name="一级业务域",
                            field_value=domain,
                        ),
                    )
                )
        except Exception as error:
            logger.warning("v2_live_filtered_query_failed error_type=%s", type(error).__name__)
            return RetrievalResult(QuestionScope.TEMPORARY_ERROR, reason="知识表查询暂时不可用")

        # Every filtered live route is only a retrieval lead, never a stop
        # condition.  The complete real-time table is still read below before
        # one semantic decision is made.  This prevents a convenient keyword
        # hit from hiding a condition, exception, or sibling rule elsewhere.
        focused_candidates = self._merge_live_candidates(
            candidates for _, candidates in stages
        )

        try:
            all_records = self.client.query_all_records(self.base_id, self.table_id)
            all_candidates = self._parse_records(self._records(all_records), now)
        except Exception as error:
            logger.warning("v2_live_full_scan_failed error_type=%s", type(error).__name__)
            return RetrievalResult(QuestionScope.TEMPORARY_ERROR, reason="知识表查询暂时不可用")

        # One semantic decision over the broadest candidate set.  Keep the
        # focused candidates as well: a filtered service may expose a newer
        # record before the full-table endpoint has indexed it.
        candidates = self._merge_live_candidates((focused_candidates, all_candidates))
        try:
            selected = self._select_live_stage(
                semantic_question,
                candidates,
                analyze,
                scopes,
                allow_partial=True,
                allow_unique_fallback=len(candidates) == 1,
            )
        except Exception as error:
            logger.warning(
                "v2_live_full_scan_selection_failed error_type=%s error=%s",
                type(error).__name__,
                str(error).replace("\n", " ")[:300],
            )
            # Do not turn several unverified candidates into an employee-facing
            # answer.  Preserve a small, relevant set for the existing human
            # handoff instead, so the owner can verify the rule from its source.
            handoff_evidence = focused_candidates or candidates
            return RetrievalResult(
                QuestionScope.TEMPORARY_ERROR,
                tuple(handoff_evidence[:5]),
                reason="语义检索暂时不可用",
            )
        if selected is not None:
            logger.info(
                "v2_live_retrieval stage=merged_full_scan focused_candidates=%d "
                "candidates=%d selected_ids=%s latency_ms=%d",
                len(focused_candidates),
                len(candidates),
                ",".join(item.knowledge_id for item in selected),
                round((time.perf_counter() - started_at) * 1000),
            )
            return RetrievalResult(
                QuestionScope.BUSINESS_MATCHED,
                tuple(self._attach_interpretations(selected)),
            )
        if QuestionScope.BUSINESS_GAP in scopes or QuestionScope.BUSINESS_MATCHED in scopes:
            return RetrievalResult(QuestionScope.BUSINESS_GAP, reason="公司业务知识不足")
        if scopes and all(scope is QuestionScope.GENERAL_CHAT for scope in scopes):
            return RetrievalResult(QuestionScope.GENERAL_CHAT)
        return RetrievalResult(QuestionScope.BUSINESS_GAP, reason="问题范围不明确")

    def _query_live_records(
        self,
        now: datetime,
        *,
        query: str,
        field_name: str = "",
        field_value: str = "",
    ) -> list[Evidence]:
        operands: list[dict[str, object]] = [
            {"operator": "eq", "operands": [self.fields["发布状态"], "已发布"]}
        ]
        if field_name and field_value:
            field_id = self.fields.get(field_name)
            if field_id:
                operands.append({"operator": "eq", "operands": [field_id, field_value]})
        payload = self.client.query_records(
            base_id=self.base_id,
            table_id=self.table_id,
            field_ids=list(self.fields.values()),
            filters={"operator": "and", "operands": operands},
            query=query,
            limit=100,
        )
        return self._parse_records(self._records(payload), now)

    @staticmethod
    def _merge_live_candidates(
        candidate_groups: Iterable[Sequence[Evidence]],
    ) -> list[Evidence]:
        """Keep all distinct live-query candidates for one semantic decision."""

        merged: list[Evidence] = []
        seen_ids: set[str] = set()
        for candidates in candidate_groups:
            for item in candidates:
                if item.knowledge_id not in seen_ids:
                    merged.append(item)
                    seen_ids.add(item.knowledge_id)
        return merged

    def _select_live_stage(
        self,
        question: str,
        candidates: list[Evidence],
        analyze: Callable[..., EvidenceSelection],
        scopes: list[QuestionScope],
        *,
        allow_partial: bool = False,
        allow_unique_fallback: bool = False,
    ) -> list[Evidence] | None:
        if not candidates:
            return None
        explicit_mappings = _question_mapping_candidates(question, candidates)
        if explicit_mappings:
            return explicit_mappings
        try:
            selected_ids, stage_scopes = self._analyze_candidate_scope(
                question,
                candidates,
                analyze,
            )
        except Exception:
            # The live text query is already a server-side relevance filter.
            # Only its single-result question stage is safe to use without a
            # semantic verdict; never guess among multiple candidates.
            if allow_unique_fallback and len(candidates) == 1:
                logger.warning(
                    "v2_live_unique_candidate_fallback knowledge_id=%s",
                    candidates[0].knowledge_id,
                )
                return candidates
            raise
        scopes.extend(stage_scopes)
        if not selected_ids and allow_partial:
            partial_analyze = getattr(self.selector, "analyze_evidence_partial", None)
            if callable(partial_analyze):
                try:
                    selected_ids, partial_scopes = self._analyze_candidate_scope(
                        question,
                        candidates,
                        partial_analyze,
                    )
                    scopes.extend(partial_scopes)
                    if selected_ids:
                        logger.info(
                            "v2_live_partial_scope_retry selected_ids=%s",
                            ",".join(selected_ids),
                        )
                except Exception as error:
                    logger.warning(
                        "v2_live_partial_scope_retry_failed error_type=%s",
                        type(error).__name__,
                    )
        by_id = {item.knowledge_id: item for item in candidates}
        selected = [by_id[item_id] for item_id in selected_ids if item_id in by_id]
        if len(selected) > 5:
            try:
                merged = analyze(question, selected, limit=5)
                merged_ids = (*merged.knowledge_ids, *merged.conflict_ids)
                selected = [by_id[item_id] for item_id in merged_ids if item_id in by_id]
            except Exception as error:
                logger.warning("v2_live_merge_failed error_type=%s", type(error).__name__)
                selected = selected[:5]
        return selected or None

    def _search_legacy(self, question: str, now: datetime) -> list[Evidence]:
        status_field = self.fields["发布状态"]
        filters = {
            "operator": "and",
            "operands": [
                {
                    "operator": "eq",
                    "operands": [status_field, "已发布"],
                }
            ],
        }
        payload = self.client.query_records(
            base_id=self.base_id,
            table_id=self.table_id,
            field_ids=list(self.fields.values()),
            filters=filters,
            query=question,
            limit=self.limit,
        )
        records = self._records(payload)
        evidence = self._parse_records(records, now)
        if evidence:
            return evidence

        fallback = self.client.query_records(
            base_id=self.base_id,
            table_id=self.table_id,
            field_ids=list(self.fields.values()),
            filters=filters,
            query="",
            limit=100,
        )
        candidates = self._rank_fallback(question, self._records(fallback), now)
        return candidates

    def _search_full_table(self, question: str, now: datetime) -> list[Evidence]:
        started_at = time.perf_counter()
        records = self._full_snapshot_records()
        candidates = self._parse_records(records, now)
        selected = self._select_in_batches(question, candidates)
        explicit_mappings = _question_mapping_candidates(question, candidates)
        if explicit_mappings:
            logger.info(
                "full_table_exact_question_mapping selected_ids=%s",
                ",".join(item.knowledge_id for item in explicit_mappings),
            )
            return self._attach_interpretations(explicit_mappings)
        selected_ids = {item.knowledge_id for item in selected}
        selected.extend(
            item
            for item in explicit_mappings
            if item.knowledge_id not in selected_ids
        )
        elapsed_ms = round((time.perf_counter() - started_at) * 1000)
        logger.info(
            "full_table_retrieval candidates=%d batches=%d exact_mappings=%d selected_ids=%s latency_ms=%d",
            len(candidates),
            (len(candidates) + self.selector_batch_size - 1)
            // self.selector_batch_size,
            len(explicit_mappings),
            ",".join(item.knowledge_id for item in selected),
            elapsed_ms,
        )
        return self._attach_interpretations(selected)

    def _retrieve_full_table(
        self,
        question: str,
        now: datetime,
        context: Sequence[ConversationTurn],
    ) -> RetrievalResult:
        started_at = time.perf_counter()
        records = self._full_snapshot_records()
        candidates = self._parse_records(records, now)
        explicit_mappings = _question_mapping_candidates(question, candidates)
        if explicit_mappings:
            logger.info(
                "full_table_exact_question_mapping selected_ids=%s",
                ",".join(item.knowledge_id for item in explicit_mappings),
            )
            return RetrievalResult(
                QuestionScope.BUSINESS_MATCHED,
                tuple(self._attach_interpretations(explicit_mappings)),
            )
        semantic_question = self._contextual_question(question, context)
        cached_ids = self._cached_selection(semantic_question, candidates)
        if cached_ids:
            by_id = {item.knowledge_id: item for item in candidates}
            cached_evidence = [by_id[item_id] for item_id in cached_ids if item_id in by_id]
            if cached_evidence:
                logger.info(
                    "full_table_selection_cache_hit selected_ids=%s",
                    ",".join(item.knowledge_id for item in cached_evidence),
                )
                return RetrievalResult(
                    QuestionScope.BUSINESS_MATCHED,
                    tuple(self._attach_interpretations(cached_evidence)),
                )
        analyze = getattr(self.selector, "analyze_evidence", None)
        if not candidates or not callable(analyze):
            if explicit_mappings:
                return RetrievalResult(
                    QuestionScope.BUSINESS_MATCHED,
                    tuple(self._attach_interpretations(explicit_mappings)),
                )
            return RetrievalResult(
                QuestionScope.TEMPORARY_ERROR,
                reason="语义检索暂时不可用",
            )

        routed_candidates: list[Evidence] | None = None
        route_decision: DomainRoutingDecision | None = None
        route = self._route_candidates(semantic_question, candidates)
        if route is not None:
            route_decision, routed_candidates = route
            if route_decision.question_scope is QuestionScope.GENERAL_CHAT:
                if explicit_mappings:
                    return RetrievalResult(
                        QuestionScope.BUSINESS_MATCHED,
                        tuple(self._attach_interpretations(explicit_mappings)),
                    )
                logger.info(
                    "domain_route_general_chat confidence=%.2f candidates=%d",
                    route_decision.confidence,
                    len(candidates),
                )
                return RetrievalResult(QuestionScope.GENERAL_CHAT)

        scan_candidates = routed_candidates or candidates
        selected_ids: list[str] = []
        scopes: list[QuestionScope] = []
        try:
            selected_ids, scopes = self._analyze_candidate_scope(
                semantic_question,
                scan_candidates,
                analyze,
            )
        except Exception as error:
            logger.warning(
                "full_table_ai_scope_failed error_type=%s",
                type(error).__name__,
            )
            if explicit_mappings:
                return RetrievalResult(
                    QuestionScope.BUSINESS_MATCHED,
                    tuple(explicit_mappings),
                )
            local_matches = self._rank_fallback(semantic_question, records, now)
            if local_matches:
                logger.info(
                    "full_table_local_fallback selected_ids=%s",
                    ",".join(item.knowledge_id for item in local_matches),
                )
                return RetrievalResult(
                    QuestionScope.BUSINESS_MATCHED,
                    tuple(self._attach_interpretations(local_matches)),
                )
            return RetrievalResult(
                QuestionScope.TEMPORARY_ERROR,
                reason="语义检索暂时不可用",
            )

        if (
            not selected_ids
            and routed_candidates is not None
            and route_decision is not None
            and route_decision.question_scope is QuestionScope.BUSINESS_MATCHED
            and _is_business_overview_request(question)
        ):
            overview_question = (
                f"{semantic_question}\n\n"
                "概览检索：用户明确在询问这个业务域的介绍或全貌。"
                "请从已确认业务域的候选中选择 2 到 5 条能代表流程、职责或发展路径的知识；"
                "只使用候选中已有的公开规则，不要补充个人判断或未提供的具体标准。"
            )
            try:
                selected_ids, overview_scopes = self._analyze_candidate_scope(
                    overview_question,
                    routed_candidates,
                    analyze,
                )
                scopes.extend(overview_scopes)
                logger.info(
                    "business_overview_retry domains=%s selected_ids=%s",
                    ",".join(route_decision.domains),
                    ",".join(selected_ids),
                )
            except Exception as error:
                logger.warning(
                    "business_overview_retry_failed error_type=%s",
                    type(error).__name__,
                )

        if routed_candidates is not None and not selected_ids:
            logger.info(
                "domain_route_full_table_fallback routed_candidates=%d total_candidates=%d",
                len(routed_candidates),
                len(candidates),
            )
            scan_candidates = candidates
            try:
                selected_ids, scopes = self._analyze_candidate_scope(
                    semantic_question,
                    scan_candidates,
                    analyze,
                )
            except Exception as error:
                logger.warning(
                    "full_table_ai_scope_fallback_failed error_type=%s",
                    type(error).__name__,
                )
                if explicit_mappings:
                    return RetrievalResult(
                        QuestionScope.BUSINESS_MATCHED,
                        tuple(explicit_mappings),
                    )
                local_matches = self._rank_fallback(semantic_question, records, now)
                if local_matches:
                    logger.info(
                        "full_table_local_fallback selected_ids=%s",
                        ",".join(item.knowledge_id for item in local_matches),
                    )
                    return RetrievalResult(
                        QuestionScope.BUSINESS_MATCHED,
                        tuple(self._attach_interpretations(local_matches)),
                    )
                return RetrievalResult(
                    QuestionScope.TEMPORARY_ERROR,
                    reason="语义检索暂时不可用",
                )

        by_id = {item.knowledge_id: item for item in candidates}
        selected = [by_id[item_id] for item_id in selected_ids if item_id in by_id]
        if len(selected) > 5:
            try:
                merged = analyze(semantic_question, selected, limit=5)
                merged_ids = (*merged.knowledge_ids, *merged.conflict_ids)
                selected = [by_id[item_id] for item_id in merged_ids if item_id in by_id]
            except Exception as error:
                logger.warning(
                    "full_table_ai_scope_merge_failed error_type=%s",
                    type(error).__name__,
                )
                selected = selected[:5]

        selected_id_set = {item.knowledge_id for item in selected}
        selected.extend(
            item
            for item in explicit_mappings
            if item.knowledge_id not in selected_id_set
        )
        elapsed_ms = round((time.perf_counter() - started_at) * 1000)
        logger.info(
            "full_table_scope_retrieval candidates=%d batches=%d scope_count=%d selected_ids=%s latency_ms=%d",
            len(scan_candidates),
            (len(scan_candidates) + self.selector_batch_size - 1)
            // self.selector_batch_size,
            len(scopes),
            ",".join(item.knowledge_id for item in selected),
            elapsed_ms,
        )
        if selected:
            self._cache_selection(semantic_question, selected)
            return RetrievalResult(
                QuestionScope.BUSINESS_MATCHED,
                tuple(self._attach_interpretations(selected)),
            )
        if QuestionScope.BUSINESS_GAP in scopes or QuestionScope.BUSINESS_MATCHED in scopes:
            return RetrievalResult(
                QuestionScope.BUSINESS_GAP,
                reason="公司业务知识不足",
            )
        if scopes and all(scope is QuestionScope.GENERAL_CHAT for scope in scopes):
            return RetrievalResult(QuestionScope.GENERAL_CHAT)
        return RetrievalResult(
            QuestionScope.BUSINESS_GAP,
            reason="问题范围不明确",
        )

    def _route_candidates(
        self,
        question: str,
        candidates: list[Evidence],
    ) -> tuple[DomainRoutingDecision, list[Evidence]] | None:
        if not self.domain_routing_enabled or len(candidates) <= self.selector_batch_size:
            return None
        route_domains = getattr(self.selector, "route_domains", None)
        if not callable(route_domains):
            return None
        profiles = self._domain_profiles(candidates)
        specific_domains = [name for name in profiles if name != "公司通用"]
        if len(specific_domains) < 2:
            return None
        try:
            decision = route_domains(question, profiles)
        except Exception as error:
            logger.warning(
                "domain_route_failed error_type=%s",
                type(error).__name__,
            )
            return None
        if decision.confidence < self.domain_routing_min_confidence:
            logger.info(
                "domain_route_low_confidence confidence=%.2f threshold=%.2f",
                decision.confidence,
                self.domain_routing_min_confidence,
            )
            return None
        named_domains = _explicitly_named_domains(question, profiles)
        if (
            decision.question_scope is QuestionScope.GENERAL_CHAT
            and named_domains
        ):
            logger.info(
                "domain_route_named_domain_override model_scope=%s domains=%s",
                decision.question_scope.value,
                ",".join(named_domains),
            )
            decision = DomainRoutingDecision(
                named_domains,
                QuestionScope.BUSINESS_MATCHED,
                decision.confidence,
            )
        if decision.question_scope is QuestionScope.GENERAL_CHAT:
            return decision, []
        if decision.question_scope is not QuestionScope.BUSINESS_MATCHED:
            return None
        selected_domains = set(decision.domains)
        selected_domains.discard("公司通用")
        if not selected_domains:
            logger.info(
                "domain_route_company_common_only_falls_back_to_full_table"
            )
            return None
        selected_domains.add("公司通用")
        routed = [
            item for item in candidates if item.business_domain in selected_domains
        ]
        if not routed or len(routed) == len(candidates):
            return None
        logger.info(
            "domain_route_applied domains=%s confidence=%.2f routed_candidates=%d total_candidates=%d",
            ",".join(decision.domains),
            decision.confidence,
            len(routed),
            len(candidates),
        )
        return decision, routed

    @staticmethod
    def _domain_profiles(
        candidates: Sequence[Evidence],
    ) -> dict[str, tuple[str, ...]]:
        profiles: dict[str, list[str]] = {}
        for item in candidates:
            domain = item.business_domain or "公司通用"
            intents = profiles.setdefault(domain, [])
            intent = item.intent.strip()
            topic = item.business_topic.strip() or "未分类"
            topic_intent = f"{topic}：{intent}" if intent else topic
            if topic_intent and topic_intent not in intents and len(intents) < 12:
                intents.append(topic_intent)
        return {domain: tuple(intents) for domain, intents in profiles.items()}

    def _analyze_candidate_scope(
        self,
        question: str,
        candidates: list[Evidence],
        analyze: Callable[..., EvidenceSelection],
    ) -> tuple[list[str], list[QuestionScope]]:
        selected_ids: list[str] = []
        scopes: list[QuestionScope] = []
        batches = [
            candidates[offset : offset + self.selector_batch_size]
            for offset in range(0, len(candidates), self.selector_batch_size)
        ]

        def analyze_batch(batch: list[Evidence]) -> EvidenceSelection:
            return analyze(question, batch, limit=5)

        if (
            len(batches) > 1
            and bool(getattr(self.selector, "supports_parallel_calls", False))
        ):
            with ThreadPoolExecutor(
                max_workers=min(MAX_PARALLEL_SELECTOR_WORKERS, len(batches)),
                thread_name_prefix="mumu-evidence-scan",
            ) as executor:
                decisions = list(executor.map(analyze_batch, batches))
        else:
            decisions = [analyze_batch(batch) for batch in batches]

        for decision in decisions:
            scopes.append(decision.question_scope)
            for knowledge_id in (*decision.knowledge_ids, *decision.conflict_ids):
                if knowledge_id not in selected_ids:
                    selected_ids.append(knowledge_id)
        return selected_ids, scopes

    def _selection_cache_key(self, question: str) -> tuple[str, float]:
        with self._snapshot_lock:
            snapshot_version = self._snapshot_loaded_at
        return question, snapshot_version

    def _cached_selection(
        self,
        question: str,
        candidates: Sequence[Evidence],
    ) -> tuple[str, ...]:
        key = self._selection_cache_key(question)
        with self._selection_cache_lock:
            selected_ids = self._selection_cache.get(key)
            if selected_ids is None:
                return ()
            available_ids = {item.knowledge_id for item in candidates}
            if not all(item_id in available_ids for item_id in selected_ids):
                self._selection_cache.pop(key, None)
                return ()
            self._selection_cache.move_to_end(key)
            return selected_ids

    def _cache_selection(
        self,
        question: str,
        evidence: Sequence[Evidence],
    ) -> None:
        selected_ids = tuple(dict.fromkeys(item.knowledge_id for item in evidence))
        if not selected_ids:
            return
        key = self._selection_cache_key(question)
        with self._selection_cache_lock:
            self._selection_cache[key] = selected_ids
            self._selection_cache.move_to_end(key)
            while len(self._selection_cache) > SELECTION_CACHE_CAPACITY:
                self._selection_cache.popitem(last=False)

    @staticmethod
    def _contextual_question(
        question: str,
        context: Sequence[ConversationTurn],
    ) -> str:
        if not context:
            return question
        lines = ["近期对话："]
        for turn in context:
            lines.extend((f"用户：{turn.question}", f"木木：{turn.answer}"))
        lines.append(f"当前问题：{question}")
        return "\n".join(lines)

    def _full_snapshot_records(self) -> list[object]:
        loaded_at = self.clock()
        with self._snapshot_lock:
            if (
                self._snapshot_records is not None
                and loaded_at - self._snapshot_loaded_at < self.snapshot_ttl_seconds
            ):
                return list(self._snapshot_records)
            attempts = (
                INITIAL_SNAPSHOT_ATTEMPTS
                if self._snapshot_records is None
                else 1
            )
            for attempt in range(1, attempts + 1):
                try:
                    payload = self.client.query_all_records(
                        self.base_id,
                        self.table_id,
                    )
                    records = self._records(payload)
                    break
                except Exception as error:
                    if attempt < attempts:
                        time.sleep(INITIAL_SNAPSHOT_RETRY_SECONDS)
                        continue
                    logger.warning(
                        "full_table_snapshot_refresh_failed error_type=%s "
                        "has_last_good=%s attempts=%d",
                        type(error).__name__,
                        self._snapshot_records is not None,
                        attempts,
                    )
                    return list(self._snapshot_records or [])
            self._snapshot_records = list(records)
            self._snapshot_loaded_at = loaded_at
            logger.info(
                "full_table_snapshot_refreshed version=%d records=%d",
                int(loaded_at),
                len(records),
            )
            return list(records)

    def invalidate_snapshot(self) -> None:
        with self._snapshot_lock:
            self._snapshot_records = None
            self._snapshot_loaded_at = 0.0
        with self._selection_cache_lock:
            self._selection_cache.clear()
        with self._interpretation_snapshot_lock:
            self._interpretation_snapshot_records = None
            self._interpretation_snapshot_loaded_at = 0.0
        with self._source_links_lock:
            self._source_links = None
            self._source_links_loaded_at = 0.0

    def _source_url_map(self) -> dict[str, str]:
        """Read source-ledger URLs once per knowledge refresh window."""
        if not self.source_table_id or not self.source_fields:
            return {}
        loaded_at = self.clock()
        with self._source_links_lock:
            if (
                self._source_links is not None
                and loaded_at - self._source_links_loaded_at < self.snapshot_ttl_seconds
            ):
                return dict(self._source_links)
            try:
                payload = self.client.query_all_records(self.base_id, self.source_table_id)
                links: dict[str, str] = {}
                for record in self._records(payload):
                    if not isinstance(record, Mapping):
                        continue
                    cells = record.get("cells")
                    if not isinstance(cells, Mapping):
                        continue
                    source_id = _cell_text(cells.get(self.source_fields.get("来源编号", "")))
                    raw_url = cells.get(self.source_fields.get("来源链接", ""))
                    url = ""
                    if isinstance(raw_url, Mapping):
                        url = _cell_text(raw_url.get("link")) or _cell_text(raw_url.get("text"))
                    else:
                        url = _cell_text(raw_url)
                    if source_id and url.startswith(("https://", "http://")):
                        links[source_id] = url
                self._source_links = links
                self._source_links_loaded_at = loaded_at
            except Exception as error:
                logger.warning("source_url_lookup_failed error_type=%s", type(error).__name__)
                return dict(self._source_links or {})
            return dict(self._source_links)

    def _attach_interpretations(self, evidence: list[Evidence]) -> list[Evidence]:
        if not evidence or not self.interpretation_table_id or not self.interpretation_fields:
            return evidence
        by_knowledge_version: dict[tuple[str, str], list[InterpretationEvidence]] = {}
        interpretation_records = (
            self._live_interpretation_records()
            if self.retrieval_mode == "v2_live"
            else self._interpretation_snapshot()
        )
        for record in interpretation_records:
            parsed = self._parse_interpretation(record)
            if parsed is None:
                continue
            knowledge_ids, interpretation = parsed
            for knowledge_id in knowledge_ids:
                by_knowledge_version.setdefault(
                    (knowledge_id, interpretation.version),
                    [],
                ).append(interpretation)
        for key, interpretations in by_knowledge_version.items():
            manual = [
                item
                for item in interpretations
                if not item.interpretation_id.startswith("MUMU-I-")
            ]
            if manual:
                by_knowledge_version[key] = manual
        return [
            replace(
                item,
                interpretations=tuple(
                    by_knowledge_version.get((item.knowledge_id, item.version), ())
                ),
            )
            for item in evidence
        ]

    def _live_interpretation_records(self) -> list[object]:
        try:
            payload = self.client.query_all_records(
                self.base_id,
                self.interpretation_table_id,
            )
            return self._records(payload)
        except Exception as error:
            logger.warning("v2_live_interpretation_query_failed error_type=%s", type(error).__name__)
            return []

    def _interpretation_snapshot(self) -> list[object]:
        loaded_at = self.clock()
        with self._interpretation_snapshot_lock:
            if (
                self._interpretation_snapshot_records is not None
                and loaded_at - self._interpretation_snapshot_loaded_at
                < self.snapshot_ttl_seconds
            ):
                return list(self._interpretation_snapshot_records)
            try:
                payload = self.client.query_all_records(
                    self.base_id,
                    self.interpretation_table_id,
                )
                records = self._records(payload)
            except Exception as error:
                logger.warning(
                    "interpretation_snapshot_refresh_failed error_type=%s has_last_good=%s",
                    type(error).__name__,
                    self._interpretation_snapshot_records is not None,
                )
                return list(self._interpretation_snapshot_records or [])
            self._interpretation_snapshot_records = list(records)
            self._interpretation_snapshot_loaded_at = loaded_at
            return list(records)

    def _parse_interpretation(
        self,
        record: object,
    ) -> tuple[tuple[str, ...], InterpretationEvidence] | None:
        if not isinstance(record, Mapping):
            return None
        cells = record.get("cells")
        if not isinstance(cells, Mapping):
            return None

        def cell(name: str) -> object:
            field_id = self.interpretation_fields.get(name)
            return cells.get(field_id) if field_id else None

        if _cell_text(cell("发布状态")) != "已发布":
            return None
        knowledge_ids = tuple(
            dict.fromkeys(
                item.strip()
                for item in re.split(
                    r"[\s,，;；]+",
                    _cell_text(cell("关联知识编号")),
                )
                if item.strip()
            )
        )
        interpretation_id = _cell_text(cell("解读编号"))
        if not knowledge_ids or not interpretation_id:
            return None
        interpretation = InterpretationEvidence(
            interpretation_id=interpretation_id,
            employee_concern=_cell_text(cell("员工关切")),
            rationale=_cell_text(cell("背后逻辑")),
            employee_value=_cell_text(cell("对员工的意义")),
            employee_cost=_cell_text(cell("员工成本与限制")),
            recommended_response=_cell_text(cell("推荐表达")),
            boundaries=_lines(cell("表达边界")),
            basis=_cell_text(cell("依据等级")),
            audience=_cell_text(cell("适用对象")),
            source_label=_cell_text(cell("解读来源")),
            source_section=_cell_text(cell("来源章节")),
            version=_cell_text(cell("关联制度版本")),
        )
        if not interpretation.rationale or not interpretation.recommended_response:
            return None
        return knowledge_ids, interpretation

    def _select_in_batches(
        self,
        question: str,
        candidates: list[Evidence],
    ) -> list[Evidence]:
        if self.selector is None or not candidates:
            return []
        selected_ids: list[str] = []
        try:
            for offset in range(0, len(candidates), self.selector_batch_size):
                batch = candidates[offset : offset + self.selector_batch_size]
                for knowledge_id in self.selector.select_evidence(
                    question,
                    batch,
                    limit=5,
                ):
                    if knowledge_id not in selected_ids:
                        selected_ids.append(knowledge_id)
        except Exception as error:
            logger.warning(
                "full_table_ai_selection_failed error_type=%s",
                type(error).__name__,
            )
            return []
        by_id = {item.knowledge_id: item for item in candidates}
        selected = [by_id[item_id] for item_id in selected_ids if item_id in by_id]
        if len(selected) <= 5:
            return selected
        try:
            final_ids = self.selector.select_evidence(question, selected, limit=5)
        except Exception as error:
            logger.warning(
                "full_table_ai_merge_failed error_type=%s",
                type(error).__name__,
            )
            return []
        selected_by_id = {item.knowledge_id: item for item in selected}
        return [
            selected_by_id[item_id]
            for item_id in final_ids
            if item_id in selected_by_id
        ][:5]

    @staticmethod
    def _records(payload: object) -> list[object]:
        if not isinstance(payload, Mapping):
            return []
        data = payload.get("data")
        if not isinstance(data, Mapping):
            return []
        records = data.get("records")
        return records if isinstance(records, list) else []

    def _parse_records(self, records: list[object], now: datetime) -> list[Evidence]:
        evidence: list[Evidence] = []
        for record in records:
            parsed = self._parse_record(record, now)
            if parsed is not None:
                evidence.append(parsed)
        return evidence

    def _rank_fallback(
        self,
        question: str,
        records: list[object],
        now: datetime,
    ) -> list[Evidence]:
        ranked: list[tuple[float, Evidence]] = []
        for record in records:
            parsed = self._parse_record(record, now)
            if parsed is None or not isinstance(record, Mapping):
                continue
            cells = record.get("cells")
            if not isinstance(cells, Mapping):
                continue
            texts = (
                parsed.intent,
                parsed.answer,
                *_questions(cells.get(self.fields["常见问法"])),
            )
            score = max(_similarity(question, text) for text in texts)
            ranked.append((score, parsed))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked:
            return []
        if ranked[0][0] >= 0.68 and (
            len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 0.05
        ):
            return [ranked[0][1]]
        if self.selector is not None:
            candidates = [item[1] for item in ranked]
            try:
                selected_ids = self.selector.select_evidence(
                    question,
                    candidates,
                    limit=5,
                )
            except Exception:
                return []
            by_id = {item.knowledge_id: item for item in candidates}
            return [by_id[item_id] for item_id in selected_ids if item_id in by_id]
        if ranked[0][0] < 0.52:
            return []
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.05:
            return []
        return [ranked[0][1]]

    def _parse_record(self, record: object, now: datetime) -> Evidence | None:
        if not isinstance(record, Mapping):
            return None
        cells = record.get("cells")
        if not isinstance(cells, Mapping):
            return None

        def cell(name: str) -> object:
            return cells.get(self.fields[name])

        try:
            effective_from = _date(cell("生效时间"))
            effective_to = _date(cell("失效时间"))
        except ValueError:
            return None
        if effective_from is not None and now < effective_from:
            return None
        if effective_to is not None and now > effective_to:
            return None
        if _cell_text(cell("发布状态")) != "已发布":
            return None
        if not _cell_text(cell("入库确认人")):
            return None

        required = {
            "knowledge_id": _cell_text(cell("知识编号")),
            "answer": _cell_text(cell("标准答案")),
            "source_id": _cell_text(cell("来源编号")),
            "source_section": _cell_text(cell("来源章节")),
        }
        if not all(required.values()):
            return None
        return Evidence(
            knowledge_id=required["knowledge_id"],
            intent=_cell_text(cell("用户意图")),
            answer=required["answer"],
            source_id=required["source_id"],
            source_section=required["source_section"],
            version=_cell_text(cell("适用制度版本")),
            risk_level=_cell_text(cell("风险等级")),
            sensitivity=_cell_text(cell("信息可见范围")),
            conditions=_lines(cell("成立条件")),
            exceptions=_lines(cell("例外情形")),
            steps=_lines(cell("办理步骤")),
            common_questions=_questions(cell("常见问法")),
            business_domain=(
                _cell_text(cell("一级业务域"))
                if "一级业务域" in self.fields
                else "公司通用"
            ) or "公司通用",
            business_topic=(
                _cell_text(cell("二级主题"))
                if "二级主题" in self.fields
                else "未分类"
            ) or "未分类",
            answer_boundary=(
                _cell_text(cell("回答边界"))
                if "回答边界" in self.fields
                else "可直接回答"
            ) or "可直接回答",
            source_url=self._source_url_map().get(required["source_id"], ""),
        )


_STORE_DIRECTORY_HINTS = (
    "地址",
    "在哪",
    "位置",
    "怎么去",
    "导航",
    "营业",
    "营运",
    "状态",
    "店型",
    "床位",
    "几张床",
    "哪个区",
    "城市",
    "实体店",
    "新店",
    "老店",
    "开业",
    "管理口径",
    "门店编号",
    "门店代码",
)


class DwsStoreDirectoryRepository:
    """Read only the sanitized store directory, never its raw source table."""

    def __init__(
        self,
        client: RecordClient,
        *,
        base_id: str,
        table_id: str,
        fields: Mapping[str, str],
        snapshot_ttl_seconds: int = 300,
        live_query: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.base_id = base_id
        self.table_id = table_id
        self.fields = dict(fields)
        self.snapshot_ttl_seconds = snapshot_ttl_seconds
        self.live_query = live_query
        self.clock = clock
        self._snapshot_records: list[object] | None = None
        self._snapshot_loaded_at = 0.0
        self._snapshot_lock = Lock()

    def retrieve(self, question: str) -> RetrievalResult:
        if not self._is_directory_question(question):
            return RetrievalResult(QuestionScope.BUSINESS_GAP)
        matched = self._matched_evidence(question)
        if not matched:
            return RetrievalResult(QuestionScope.BUSINESS_GAP)
        return RetrievalResult(QuestionScope.BUSINESS_MATCHED, tuple(matched))

    def invalidate_snapshot(self) -> None:
        with self._snapshot_lock:
            self._snapshot_records = None
            self._snapshot_loaded_at = 0.0

    def _matched_evidence(self, question: str) -> list[Evidence]:
        normalized_question = _normalized(question)
        matched: list[tuple[int, Evidence]] = []
        for record in self._snapshot_records_for_directory():
            evidence = self._parse_store_record(record)
            if evidence is None:
                continue
            names = (evidence.intent, *evidence.common_questions)
            normalized_names = {
                _normalized(name)
                for name in names
                if _normalized(name)
            }
            matched_names = [
                name for name in normalized_names if name in normalized_question
            ]
            if matched_names:
                matched.append((max(map(len, matched_names)), evidence))
        matched.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in matched[:3]]

    @staticmethod
    def _is_directory_question(question: str) -> bool:
        normalized_question = _normalized(question)
        return any(_normalized(hint) in normalized_question for hint in _STORE_DIRECTORY_HINTS)

    def _snapshot_records_for_directory(self) -> list[object]:
        if self.live_query:
            try:
                payload = self.client.query_all_records(self.base_id, self.table_id)
                return DwsKnowledgeRepository._records(payload)
            except Exception as error:
                logger.warning("store_directory_live_query_failed error_type=%s", type(error).__name__)
                return []
        loaded_at = self.clock()
        with self._snapshot_lock:
            if (
                self._snapshot_records is not None
                and loaded_at - self._snapshot_loaded_at < self.snapshot_ttl_seconds
            ):
                return list(self._snapshot_records)
            try:
                payload = self.client.query_all_records(self.base_id, self.table_id)
                records = DwsKnowledgeRepository._records(payload)
            except Exception as error:
                logger.warning(
                    "store_directory_snapshot_refresh_failed error_type=%s has_last_good=%s",
                    type(error).__name__,
                    self._snapshot_records is not None,
                )
                return list(self._snapshot_records or [])
            self._snapshot_records = list(records)
            self._snapshot_loaded_at = loaded_at
            return list(records)

    def _parse_store_record(self, record: object) -> Evidence | None:
        if not isinstance(record, Mapping):
            return None
        cells = record.get("cells")
        if not isinstance(cells, Mapping):
            return None

        def value(name: str) -> str:
            field_id = self.fields.get(name)
            cell = cells.get(field_id) if field_id else None
            text = _cell_text(cell)
            if text:
                return text
            if isinstance(cell, (int, float)):
                return str(cell)
            return ""

        store_name = value("门店名称")
        if not store_name:
            return None
        alias = value("门店别名")
        managed_store = value("对应管理门店")
        details = (
            ("门店编号", value("门店编号")),
            ("城市", value("城市")),
            ("行政区", value("行政区")),
            ("营运状态", value("营运状态")),
            ("店型", value("店型")),
            ("门店分级", value("门店分级")),
            ("开业时间", value("开业时间")),
            ("床位数", value("床位数")),
            ("地址", value("地址")),
            ("对应管理门店", value("对应管理门店")),
        )
        answer = "；".join(f"{label}：{item}" for label, item in details if item)
        if not answer:
            return None
        common_questions = [f"{store_name}在哪", f"{store_name}还在营业吗"]
        if alias and _normalized(alias) != _normalized(store_name):
            common_questions.extend((alias, f"{alias}对应哪家门店"))
        if managed_store and _normalized(managed_store) not in {
            _normalized(store_name),
            _normalized(alias),
        }:
            common_questions.extend(
                (managed_store, f"{managed_store}对应哪些门店")
            )
        return Evidence(
            knowledge_id=f"STORE-DIR-{_normalized(store_name)}",
            intent=store_name,
            answer=f"{store_name}：{answer}。",
            source_id="STORE-S001",
            source_section="门店信息宽表 / 门店基础信息（实时读取，安全字段）",
            version="directory",
            risk_level="普通",
            sensitivity="限内部",
            common_questions=tuple(common_questions),
            business_domain="公司通用",
        )


@dataclass(frozen=True)
class LiveKnowledgeSource:
    name: str
    workspace_id: str
    business_domain: str = "公司通用"
    business_topic: str = "实时知识库"
    answer_boundary: str = "以知识库当前原文为准"


class DwsStoreGuideRepository:
    """Read selected current knowledge-base documents on demand."""

    _MAX_DOCUMENTS = 3
    _MAX_SEARCH_QUERIES = 6
    # Keep broad semantic recall without multiplying requests when several
    # workspaces are selected for the same question.
    _TOTAL_SEARCH_QUERY_BUDGET = 10
    _MAX_DOCUMENT_CHARS = 6000
    _RETIRED_DOCUMENT_TITLE = re.compile(r"(?:弃用|历史|旧版)")
    _KNOWN_CITIES = (
        "上海", "深圳", "广州", "北京", "杭州", "成都", "武汉", "南京",
        "苏州", "重庆", "西安", "天津",
    )
    _QUESTION_FILLERS = re.compile(
        r"会员|客户|门店|现在|目前|请问|麻烦|应该|怎么|如何|怎么办|处理|一下|"
        r"这个|那个|有没有|能不能|可不可以|是否|什么|哪些|多少|的|是|了|吗|呢"
    )

    def __init__(
        self,
        client: StoreGuideClient,
        *,
        workspace_id: str = "",
        sources: Sequence[LiveKnowledgeSource] = (),
    ) -> None:
        configured_sources = tuple(sources)
        if not configured_sources and workspace_id.strip():
            configured_sources = (
                LiveKnowledgeSource(
                    name="门店指引",
                    workspace_id=workspace_id,
                    business_domain="门店运营",
                    business_topic="门店指引",
                    answer_boundary="以门店指引当前内容为准；平台、价格、审批和时效以实时页面为准",
                ),
            )
        if not configured_sources:
            raise ValueError("至少需要配置一个实时知识库")
        if any(not source.name.strip() or not source.workspace_id.strip() for source in configured_sources):
            raise ValueError("实时知识库名称和 ID 不能为空")
        self.client = client
        self.sources = configured_sources
        self.workspace_id = configured_sources[0].workspace_id
        self.live_source_names = tuple(source.name for source in configured_sources)

    def retrieve(
        self,
        question: str,
        source_names: Sequence[str] = (),
        search_policy: str = "single",
    ) -> RetrievalResult:
        queries = self._search_queries(question)
        if not queries:
            return RetrievalResult(QuestionScope.BUSINESS_GAP)
        selected_names = tuple(dict.fromkeys(source_names))
        sources_by_name = {source.name: source for source in self.sources}
        sources = (
            tuple(sources_by_name[name] for name in selected_names if name in sources_by_name)
            if selected_names
            else self.sources
        )
        if not sources:
            return RetrievalResult(QuestionScope.BUSINESS_GAP)
        if search_policy not in {"single", "cross_domain", "fallback"}:
            search_policy = "single"
        if search_policy == "single":
            return self._retrieve_sources(question, queries, sources[:1])
        if search_policy == "fallback":
            # Probe one business library at a time.  A valid result from the
            # primary candidate ends the search and keeps citations focused.
            for source in sources:
                result = self._retrieve_sources(question, queries, (source,))
                if result.scope is QuestionScope.BUSINESS_MATCHED:
                    return result
            return RetrievalResult(QuestionScope.BUSINESS_GAP)
        return self._retrieve_sources(question, queries, sources)

    def _retrieve_sources(
        self,
        question: str,
        queries: Sequence[str],
        sources: Sequence[LiveKnowledgeSource],
    ) -> RetrievalResult:
        query_limit = min(
            len(queries),
            max(1, self._TOTAL_SEARCH_QUERY_BUDGET // len(sources)),
        )
        bounded_queries = queries[:query_limit]
        try:
            with ThreadPoolExecutor(max_workers=min(3, len(sources))) as executor:
                results = list(
                    executor.map(
                        lambda source: self._retrieve_source(
                            question,
                            bounded_queries,
                            source,
                        ),
                        sources,
                    )
                )
        except Exception as error:
            logger.warning(
            "live_knowledge_search_failed error_type=%s",
                type(error).__name__,
            )
            return RetrievalResult(QuestionScope.BUSINESS_GAP)
        evidence = [item for result in results for item in result][:6]
        if not evidence:
            return RetrievalResult(QuestionScope.BUSINESS_GAP)
        logger.info(
            "live_knowledge_retrieval queries=%d source_count=%d selected_documents=%d sources=%s",
            len(bounded_queries),
            len(sources),
            len(evidence),
            ",".join(item.source_id for item in evidence),
        )
        return RetrievalResult(QuestionScope.BUSINESS_MATCHED, tuple(evidence))

    def _retrieve_source(
        self,
        question: str,
        queries: Sequence[str],
        source: LiveKnowledgeSource,
    ) -> list[Evidence]:
        found_documents: list[tuple[int, str, object]] = []
        for index, query in enumerate(queries):
            payload = self.client.search_wiki_nodes(
                source.workspace_id,
                query,
                limit=self._MAX_DOCUMENTS * 2,
            )
            found = payload.get("documents")
            if isinstance(found, list) and found:
                found_documents.extend((index, query, document) for document in found)
        documents = self._rank_documents(question, found_documents)
        return self._read_documents(documents, source)

    @classmethod
    def _search_queries(cls, question: str) -> tuple[str, ...]:
        # Test prefixes are not part of the employee's actual issue and reduce
        # enterprise-search recall when a message is replayed for verification.
        cleaned = re.sub(r"【[^】]{0,80}】", "", question)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned or cleaned in {"你好", "在吗", "谢谢", "好的", "ok", "OK"}:
            return ()
        queries = [cleaned[:180]]
        core = cls._core_query(cleaned)
        if core and core != cleaned:
            queries.append(core)
        # A document search can return a weak match for a single particle or a
        # short fragment.  Search the meaningful concepts too, then rank the
        # whole union instead of trusting whichever query happened to respond.
        terms = cls._meaningful_terms(core or cleaned)
        # Natural questions usually end with the document topic ("卡项"、
        # "退款"、"签到" etc.).  Include those compact concepts early instead
        # of exhausting the query budget on overlapping long fragments.
        compact = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", core or cleaned)
        compact_terms = sorted(
            (term for term in terms if len(_normalized(term)) == 2),
            key=lambda term: -compact.rfind(term),
        )
        queries.extend(compact_terms)
        queries.extend(term for term in terms if term not in compact_terms)
        unique = tuple(dict.fromkeys(query for query in queries if len(query) >= 2))
        return unique[: cls._MAX_SEARCH_QUERIES]

    @classmethod
    def _rank_documents(
        cls,
        question: str,
        found_documents: Sequence[tuple[int, str, object]],
    ) -> list[object]:
        """Merge every live search result and reject fragment-only false hits."""

        core = cls._core_query(question)
        terms = cls._meaningful_terms(core or question)
        question_cities = cls._mentioned_cities(question)
        by_node_id: dict[str, tuple[int, int, bool, bool, object]] = {}
        for query_index, query, document in found_documents:
            if not isinstance(document, Mapping):
                continue
            node_id = _cell_text(document.get("nodeId"))
            title = _cell_text(document.get("name"))
            if not node_id or not title or cls._RETIRED_DOCUMENT_TITLE.search(title):
                continue
            title_cities = cls._mentioned_cities(title)
            if question_cities and title_cities and question_cities.isdisjoint(title_cities):
                continue
            normalized_title = _normalized(title)
            overlaps = [term for term in terms if _normalized(term) in normalized_title]
            # Results returned only by a shortened query must prove their
            # relevance in the title.  This is what prevents a document such
            # as "退款挽留 SOP" from winning a "深圳卡项" question merely because
            # the search engine accepted an uninformative fragment.
            if query_index >= 2 and not overlaps:
                continue
            title_score = sum(min(len(_normalized(term)), 4) * 20 for term in overlaps)
            core_score = 40 if core and core in normalized_title else 0
            query_score = max(0, 20 - query_index * 3)
            score = title_score + core_score + query_score
            has_title_overlap = bool(overlaps)
            has_specific_overlap = any(len(_normalized(term)) >= 3 for term in overlaps)
            previous = by_node_id.get(node_id)
            if previous is None or score > previous[0] or (
                score == previous[0] and query_index < previous[1]
            ):
                by_node_id[node_id] = (
                    score,
                    query_index,
                    has_title_overlap,
                    has_specific_overlap,
                    document,
                )
        ranked = sorted(
            by_node_id.values(),
            key=lambda item: (-item[0], item[1], _cell_text(item[4].get("name"))),
        )
        # Once at least one document title explicitly matches the employee's
        # core concepts, it is stronger evidence than a broad full-question
        # search result with no shared concept.  Do not mix those weak leads
        # into the final answer merely to fill the document limit.
        if any(has_specific_overlap for _, _, _, has_specific_overlap, _ in ranked):
            ranked = [item for item in ranked if item[3]]
        elif any(has_overlap for _, _, has_overlap, _, _ in ranked):
            ranked = [item for item in ranked if item[2]]
        return [document for _, _, _, _, document in ranked[: cls._MAX_DOCUMENTS]]

    @classmethod
    def _mentioned_cities(cls, text: str) -> set[str]:
        return {city for city in cls._KNOWN_CITIES if city in text}

    @classmethod
    def _core_query(cls, text: str) -> str:
        core = cls._QUESTION_FILLERS.sub("", text)
        core = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", core)
        return core[:80]

    @staticmethod
    def _meaningful_terms(text: str) -> tuple[str, ...]:
        compact = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)
        candidates: list[str] = []
        for block in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", compact):
            if len(block) <= 6:
                candidates.append(block)
            for size in range(min(4, len(block)), 1, -1):
                candidates.extend(block[index : index + size] for index in range(len(block) - size + 1))
        # Prefer complete concepts first, then the final noun phrase.  The
        # latter is often the actual document topic in natural-language asks.
        unique = list(dict.fromkeys(candidate for candidate in candidates if len(candidate) >= 2))
        unique.sort(key=lambda item: (-len(item), -compact.rfind(item), item))
        return tuple(unique)

    def _read_documents(
        self,
        documents: Sequence[object],
        source: LiveKnowledgeSource,
    ) -> list[Evidence]:
        evidence: list[Evidence] = []
        seen_node_ids: set[str] = set()
        for document in documents:
            if len(evidence) >= self._MAX_DOCUMENTS:
                break
            if not isinstance(document, Mapping):
                continue
            node_id = _cell_text(document.get("nodeId"))
            title = _cell_text(document.get("name")) or source.name
            # Enterprise search can return older SOPs together with the
            # current one.  Never let a document explicitly marked retired,
            # historical, or old version participate in a live answer.
            if self._RETIRED_DOCUMENT_TITLE.search(title):
                continue
            if not node_id or node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            payload = self.client.read_doc(node_id)
            markdown = _cell_text(payload.get("markdown"))
            if not markdown:
                continue
            evidence.append(
                Evidence(
                    knowledge_id=f"LIVE-{_normalized(source.name)}-{node_id}",
                    intent=title,
                    answer=self._trim_document(markdown),
                    source_id=title,
                    source_section=f"{source.name}实时读取",
                    version="实时读取",
                    risk_level="普通",
                    sensitivity="企业内部",
                    business_domain=source.business_domain,
                    business_topic=source.business_topic,
                    answer_boundary=source.answer_boundary,
                    source_url=f"https://alidocs.dingtalk.com/i/nodes/{node_id}",
                )
            )
        return evidence

    @classmethod
    def _trim_document(cls, markdown: str) -> str:
        without_images = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", markdown)
        compact = re.sub(r"\n{3,}", "\n\n", without_images).strip()
        return compact[: cls._MAX_DOCUMENT_CHARS]


class StoreGuideAwareKnowledgeRepository:
    """Route between live documents and the existing AI-table chain."""

    def __init__(
        self,
        knowledge_repository: KnowledgeRetriever,
        store_guide: DwsStoreGuideRepository,
    ) -> None:
        self.knowledge_repository = knowledge_repository
        self.store_guide = store_guide

    def __getattr__(self, name: str) -> object:
        return getattr(self.knowledge_repository, name)

    def retrieve(
        self,
        question: str,
        now: datetime,
        context: Sequence[ConversationTurn] = (),
        source_mode: str | None = None,
        live_sources: Sequence[str] = (),
        live_search_policy: str = "single",
    ) -> RetrievalResult:
        if source_mode is None:
            guide_result = self.store_guide.retrieve(question, live_sources, live_search_policy)
            if guide_result.scope is QuestionScope.BUSINESS_MATCHED:
                return guide_result
            return self.knowledge_repository.retrieve(question, now, context)
        if source_mode not in {"realtime", "aitable", "both"}:
            source_mode = "both"
        guide_result = (
            self.store_guide.retrieve(question, live_sources, live_search_policy)
            # Legacy deployments have a single live source and did not carry
            # a source-selection result.  Preserve that path; with several
            # sources an empty selection must not fan out to every workspace.
            if source_mode in {"realtime", "both"}
            and (live_sources or len(self.store_guide.sources) == 1)
            else RetrievalResult(QuestionScope.BUSINESS_GAP)
        )
        if source_mode == "realtime":
            # A realtime route must never silently answer from a stale AI-table
            # rule when the live document search is empty or unavailable.  The
            # directory wrapper is still allowed to contribute exact store
            # facts, but all other AI-table evidence waits for an explicit
            # `both` route from the semantic router.
            directory_result = self.knowledge_repository.retrieve(
                question,
                now,
                context,
                source_mode="realtime",
            )
            directory_evidence = tuple(
                item
                for item in directory_result.evidence
                if item.knowledge_id.startswith("STORE-DIR-")
            )
            if guide_result.evidence or directory_evidence:
                return self._merge_results(
                    guide_result,
                    RetrievalResult(
                        QuestionScope.BUSINESS_MATCHED,
                        directory_evidence,
                    ),
                )
            if (
                guide_result.scope is QuestionScope.TEMPORARY_ERROR
                or directory_result.scope is QuestionScope.TEMPORARY_ERROR
            ):
                return RetrievalResult(
                    QuestionScope.TEMPORARY_ERROR,
                    reason="实时知识库查询暂时不可用",
                )
            return RetrievalResult(QuestionScope.BUSINESS_GAP, reason="实时原文无匹配")

        table_result = self.knowledge_repository.retrieve(
            question,
            now,
            context,
            source_mode=source_mode,
        )
        if source_mode == "aitable":
            return table_result
        return self._merge_results(guide_result, table_result)

    @staticmethod
    def _merge_results(
        live_result: RetrievalResult,
        table_result: RetrievalResult,
    ) -> RetrievalResult:
        evidence: list[Evidence] = []
        seen_ids: set[str] = set()
        for result in (live_result, table_result):
            for item in result.evidence:
                if item.knowledge_id not in seen_ids:
                    evidence.append(item)
                    seen_ids.add(item.knowledge_id)
        if evidence:
            return RetrievalResult(QuestionScope.BUSINESS_MATCHED, tuple(evidence))
        if (
            live_result.scope is QuestionScope.TEMPORARY_ERROR
            or table_result.scope is QuestionScope.TEMPORARY_ERROR
        ):
            return RetrievalResult(QuestionScope.TEMPORARY_ERROR, reason="知识来源查询暂时不可用")
        if (
            live_result.scope is QuestionScope.GENERAL_CHAT
            and table_result.scope is QuestionScope.GENERAL_CHAT
        ):
            return RetrievalResult(QuestionScope.GENERAL_CHAT)
        return RetrievalResult(QuestionScope.BUSINESS_GAP, reason="公司业务知识不足")

    def invalidate_snapshot(self) -> None:
        invalidate = getattr(self.knowledge_repository, "invalidate_snapshot", None)
        if callable(invalidate):
            invalidate()


class StoreAwareKnowledgeRepository:
    """Prefer exact store-directory facts, then continue to semantic knowledge retrieval."""

    def __init__(
        self,
        knowledge_repository: DwsKnowledgeRepository,
        store_directory: DwsStoreDirectoryRepository,
    ) -> None:
        self.knowledge_repository = knowledge_repository
        self.store_directory = store_directory

    def __getattr__(self, name: str) -> object:
        """Keep the wrapper transparent to existing runtime diagnostics."""
        return getattr(self.knowledge_repository, name)

    def retrieve(
        self,
        question: str,
        now: datetime,
        context: Sequence[ConversationTurn] = (),
        source_mode: str = "both",
        live_sources: Sequence[str] = (),
        live_search_policy: str = "single",
    ) -> RetrievalResult:
        del live_sources, live_search_policy
        if source_mode in {"realtime", "aitable", "both"}:
            store_result = self.store_directory.retrieve(question)
            if store_result.scope is QuestionScope.BUSINESS_MATCHED:
                return store_result
        return self.knowledge_repository.retrieve(
            question,
            now,
            context,
            source_mode=source_mode,
        )

    def invalidate_snapshot(self) -> None:
        self.knowledge_repository.invalidate_snapshot()
        self.store_directory.invalidate_snapshot()
