from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Protocol, TextIO, cast

from dingtalk_stream import (
    CallbackHandler,
    ChatbotMessage,
    Credential,
)
from mumu_kb.dws import DwsClient
from mumu_kb.models import TestCase, load_json_records
from mumu_kb.provision import load_manifest

from .config import BotConfig, ConfigError
from .case_card import CaseCardGateway, MumuCaseCardHandler
from .case_coordinator import CaseCoordinator
from .case_gateway import DwsCaseGateway
from .case_reminder import CaseDraftReminder
from .case_repository import CaseRepository
from .case_resolution import CaseResolutionService
from .case_submission import CaseSubmissionRunner, CaseSubmissionWorker
from .deepseek import DeepSeekClient
from .domain import AnswerResult
from .evaluation import NullGapRecorder, evaluate_cases
from .gaps import JsonGapRecorder
from .memory import ConversationMemory
from .interpretation_enrichment import (
    DwsInterpretationStore,
    InterpretationEnrichmentRunner,
    InterpretationEnrichmentWorker,
)
from .monitored_stream import MonitoredDingTalkStreamClient
from .reaction import RobotThinkingReactionPort, TrackedThinkingReactionPort
from .repository import (
    DwsKnowledgeRepository,
    DwsStoreDirectoryRepository,
    DwsStoreGuideRepository,
    LiveKnowledgeSource,
    StoreAwareKnowledgeRepository,
    StoreGuideAwareKnowledgeRepository,
)
from .runtime_state import StreamHealthReporter
from .service import SupportService
from .stream import MumuChatbotHandler, SessionWebhookReplyPort


class AnswerService(Protocol):
    def answer(
        self,
        user_id: str,
        question: str,
        now: datetime,
    ) -> AnswerResult: ...


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mumu-bot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ask = subparsers.add_parser("ask", help="本地运行一次木木问答")
    ask.add_argument("--question", required=True)
    ask.add_argument("--user-id", default="local-cli")
    evaluate = subparsers.add_parser("evaluate", help="运行木木离线准确率验收")
    evaluate.add_argument(
        "--tests",
        default="knowledge/employee-partnership-3.0/test-cases.json",
    )
    evaluate.add_argument(
        "--output",
        default=".runtime/evaluation-report.json",
    )
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument(
        "--offset",
        type=int,
        default=0,
        help="从第几条测试开始执行，用于分批续跑",
    )
    evaluate.add_argument(
        "--skip-semantic-judge",
        action="store_true",
        help="只使用确定性要点匹配，减少额外模型调用",
    )
    subparsers.add_parser("start", help="启动木木钉钉 Stream 服务")
    return parser


def _build_service(config: BotConfig) -> SupportService:
    manifest = load_manifest(Path(".runtime/aitable-manifest.json"))
    tables = cast(dict[str, object], manifest["tables"])
    knowledge = cast(dict[str, object], tables["知识条目"])
    fields = cast(dict[str, str], knowledge["fields"])
    source_ledger = tables.get("知识源台账")
    source_table_id = ""
    source_fields: dict[str, str] = {}
    if isinstance(source_ledger, dict):
        source_table_id = str(source_ledger.get("table_id", ""))
        source_fields = cast(dict[str, str], source_ledger.get("fields", {}))
    model = DeepSeekClient(
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_base_url,
        model=config.deepseek_model,
        skill_version=config.skill_version,
        timeout=float(config.deepseek_timeout_seconds),
    )
    interpretation_table_id = ""
    interpretation_fields: dict[str, str] = {}
    reference_table_id = ""
    reference_fields: dict[str, str] = {}
    dws_client = DwsClient(config.dws_profile)
    if config.interpretation_enabled:
        schema_version = manifest.get("schema_version")
        interpretation = tables.get("制度解读")
        if not isinstance(schema_version, int) or schema_version < 4:
            raise ConfigError("制度解读功能必须先完成 AI 表格 schema v4 迁移")
        if not isinstance(interpretation, dict):
            raise ConfigError("运行清单缺少制度解读表")
        interpretation_table_id = cast(str, interpretation["table_id"])
        interpretation_fields = cast(dict[str, str], interpretation["fields"])
    reference = tables.get("资料参考库")
    if isinstance(reference, dict):
        reference_table_id = cast(str, reference.get("table_id", ""))
        reference_fields = cast(dict[str, str], reference.get("fields", {}))
    knowledge_repository = DwsKnowledgeRepository(
        dws_client,
        base_id=cast(str, manifest["base_id"]),
        table_id=cast(str, knowledge["table_id"]),
        fields=fields,
        selector=model,
        retrieval_mode=config.retrieval_mode,
        snapshot_ttl_seconds=config.knowledge_refresh_seconds,
        selector_batch_size=config.full_scan_batch_size,
        domain_routing_enabled=config.domain_routing_enabled,
        domain_routing_min_confidence=config.domain_routing_min_confidence,
        interpretation_table_id=interpretation_table_id,
        interpretation_fields=interpretation_fields,
        reference_table_id=reference_table_id,
        reference_fields=reference_fields,
        source_table_id=source_table_id,
        source_fields=source_fields,
    )
    live_store = manifest.get("live_store_directory")
    if not isinstance(live_store, dict):
        raise ConfigError("运行清单缺少实时门店信息来源")
    store_base_id = cast(str, live_store.get("base_id", ""))
    store_table_id = cast(str, live_store.get("table_id", ""))
    store_fields = cast(dict[str, str], live_store.get("fields", {}))
    repository = knowledge_repository
    if not store_base_id or not store_table_id or not store_fields:
        raise ConfigError("实时门店信息来源配置不完整")
    repository = StoreAwareKnowledgeRepository(
        knowledge_repository,
        DwsStoreDirectoryRepository(
            dws_client,
            base_id=store_base_id,
            table_id=store_table_id,
            fields=store_fields,
            snapshot_ttl_seconds=config.knowledge_refresh_seconds,
            live_query=True,
        )
    )
    live_sources_config = manifest.get("live_knowledge_sources")
    if not isinstance(live_sources_config, list) or not live_sources_config:
        raise ConfigError("运行清单缺少实时知识库来源")
    live_sources: list[LiveKnowledgeSource] = []
    for item in live_sources_config:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        workspace_id = item.get("workspace_id")
        if not isinstance(name, str) or not isinstance(workspace_id, str):
            continue
        live_sources.append(
            LiveKnowledgeSource(
                name=name,
                workspace_id=workspace_id,
                business_domain=str(item.get("business_domain", "公司通用")),
                business_topic=str(item.get("business_topic", "实时知识库")),
                answer_boundary=str(item.get("answer_boundary", "以知识库当前原文为准")),
            )
        )
    if not live_sources:
        raise ConfigError("实时知识库来源配置不完整")
    repository = StoreGuideAwareKnowledgeRepository(
        repository,
        DwsStoreGuideRepository(
            dws_client,
            sources=live_sources,
        ),
    )
    return SupportService(
        repository=repository,
        model=model,
        gap_recorder=JsonGapRecorder(Path(".runtime/knowledge-gaps.json")),
        operator_user_id=config.operator_user_id,
        memory=ConversationMemory(
            state_path=Path(".runtime/conversation-memory.json"),
        ),
    )


def _build_case_coordinator(
    config: BotConfig,
    service: SupportService,
    card_gateway: CaseCardGateway | None = None,
) -> CaseCoordinator | None:
    if not config.case_workflow_enabled:
        return None
    manifest = load_manifest(Path(".runtime/aitable-manifest.json"))
    schema_version = manifest.get("schema_version")
    if not isinstance(schema_version, int) or schema_version < 3:
        raise ConfigError("自动案件流程必须先完成 AI 表格迁移并回读校验")
    dws_client = DwsClient(config.dws_profile)
    case_repository = CaseRepository(dws_client, manifest)
    return CaseCoordinator(
        repository=case_repository,
        gateway=DwsCaseGateway(
            dws_client,
            robot_code=config.dingtalk_robot_code,
            card_gateway=card_gateway,
        ),
        model=service.model,
        account_owner_user_id=config.owner_user_id,
        account_owner_name=config.owner_name,
        responsible_user_id=config.operator_user_id,
        responsible_name="人工负责人",
        max_messages=config.case_max_messages,
        knowledge_repository=service.repository,
    )


def _build_interpretation_runner(
    config: BotConfig,
    service: SupportService,
) -> InterpretationEnrichmentRunner | None:
    if not config.interpretation_enabled:
        return None
    manifest = load_manifest(Path(".runtime/aitable-manifest.json"))
    store = DwsInterpretationStore(DwsClient(config.dws_profile), manifest)
    return InterpretationEnrichmentRunner(
        InterpretationEnrichmentWorker(
            store=store,
            model=service.model,
            knowledge_repository=service.repository,
            batch_size=5,
        )
    )


def _run_stream(config: BotConfig, service: AnswerService) -> None:
    credential = Credential(
        config.dingtalk_client_id,
        config.dingtalk_client_secret,
    )
    reporter = StreamHealthReporter(
        Path(".runtime/stream-health.json"),
        os.environ.get("MUMU_RUNTIME_RUN_ID", f"manual-{os.getpid()}"),
    )
    reporter.starting()
    client = MonitoredDingTalkStreamClient(credential, reporter)
    card_gateway = (
        CaseCardGateway(
            access_token_provider=client.get_access_token,
            robot_code=config.dingtalk_robot_code,
            card_template_id=config.case_card_template_id,
        )
        if config.case_workflow_enabled
        else None
    )
    case_coordinator = (
        _build_case_coordinator(config, service, card_gateway)
        if isinstance(service, SupportService)
        else None
    )
    interpretation_runner = (
        _build_interpretation_runner(config, service)
        if isinstance(service, SupportService)
        else None
    )
    reaction_port = TrackedThinkingReactionPort(
        RobotThinkingReactionPort(
            access_token_provider=client.get_access_token,
            robot_code=config.dingtalk_robot_code,
        ),
        Path(".runtime/active-thinking-reactions.json"),
    )
    reaction_port.cleanup_stale()
    handler = MumuChatbotHandler(
        service=service,
        reply_port=SessionWebhookReplyPort(),
        reaction_port=reaction_port,
        allowed_user_ids=set(config.allowed_user_ids),
        operator_user_id=config.operator_user_id,
        case_coordinator=case_coordinator,
    )
    client.register_callback_handler(ChatbotMessage.TOPIC, handler)
    if (
        case_coordinator is not None
        and card_gateway is not None
        and isinstance(service, SupportService)
    ):
        draft_reminder = CaseDraftReminder(
            case_coordinator.repository.client,
        executor_user_ids=(config.operator_user_id, config.owner_user_id),
            state_path=Path(".runtime/case-draft-reminders.json"),
            knowledge_table_url=(
                "https://alidocs.dingtalk.com/i/nodes/"
                f"{case_coordinator.repository.base_id}"
            ),
        )
        resolution = CaseResolutionService(
            repository=case_coordinator.repository,
            gateway=case_coordinator.gateway,
            model=case_coordinator.model,
            max_messages=case_coordinator.max_messages,
            knowledge_repository=service.repository,
            draft_reminder=draft_reminder,
        )
        client.register_callback_handler(
            CallbackHandler.TOPIC_CARD_CALLBACK,
            MumuCaseCardHandler(
                resolution=resolution,
                card_gateway=card_gateway,
            ),
        )
        CaseSubmissionRunner(
            CaseSubmissionWorker(
                repository=case_coordinator.repository,
                resolution=resolution,
                card_gateway=card_gateway,
                draft_reminder=draft_reminder,
            )
        ).start()
    if interpretation_runner is not None:
        interpretation_runner.start()
    try:
        client.start_forever()
    finally:
        if interpretation_runner is not None:
            interpretation_runner.stop()


def _print_text(value: str, stdout: TextIO) -> None:
    try:
        print(value, file=stdout)
    except UnicodeEncodeError:
        encoding = getattr(stdout, "encoding", None) or "utf-8"
        safe_value = value.encode(encoding, errors="replace").decode(encoding)
        print(safe_value, file=stdout)


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] = os.environ,
    stdout: TextIO = sys.stdout,
    service_factory: Callable[[BotConfig], AnswerService] = _build_service,
    stream_runner: Callable[[BotConfig, AnswerService], None] = _run_stream,
) -> int:
    args = _parser().parse_args(argv)
    try:
        config = BotConfig.from_env(environ)
        service = service_factory(config)
        if args.command == "start":
            print("木木 Stream 服务已启动。", file=stdout, flush=True)
            stream_runner(config, service)
            return 0
        if args.command == "evaluate":
            test_cases = load_json_records(Path(args.tests), TestCase)
            if args.offset < 0:
                raise ConfigError("evaluate --offset 不能小于0")
            if args.limit is not None:
                if args.limit < 1:
                    raise ConfigError("evaluate --limit 必须大于0")
            stop = args.offset + args.limit if args.limit is not None else None
            test_cases = test_cases[args.offset : stop]
            evaluation_service: AnswerService = service
            if isinstance(service, SupportService):
                evaluation_service = SupportService(
                    repository=service.repository,
                    model=service.model,
                    gap_recorder=NullGapRecorder(),
                    operator_user_id=service.operator_user_id,
                )
            report = evaluate_cases(
                evaluation_service,
                test_cases,
                datetime.now().astimezone(),
                point_judge=(
                    None
                    if args.skip_semantic_judge
                    else (service.model if isinstance(service, SupportService) else None)
                ),
            )
            output_path = Path(args.output)
            report.write_json(output_path)
            report.write_markdown(output_path.with_suffix(".md"))
            print(
                "验收完成："
                f"{report.passed_cases}/{report.total_cases}完全通过，"
                f"知识要点准确率{report.knowledge_point_accuracy:.2%}，"
                f"来源匹配率{report.source_match_rate:.2%}，"
                f"转人工正确率{report.handoff_accuracy:.2%}。",
                file=stdout,
            )
            return 0 if report.passes_gate else 1
        result = service.answer(
            args.user_id,
            args.question,
            datetime.now().astimezone(),
        )
    except ConfigError as error:
        print(f"配置错误：{error}", file=stdout)
        return 2
    except Exception as error:
        print(f"运行失败：{type(error).__name__}", file=stdout)
        return 3
    _print_text(result.text, stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
