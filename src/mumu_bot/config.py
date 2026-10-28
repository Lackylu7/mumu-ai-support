from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class BotConfig:
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    skill_version: str
    dingtalk_client_id: str
    dingtalk_client_secret: str
    dingtalk_robot_code: str
    dws_profile: str
    store_guide_workspace_id: str
    operator_user_id: str
    allowed_user_ids: tuple[str, ...]
    retrieval_mode: str
    knowledge_refresh_seconds: int
    full_scan_batch_size: int
    domain_routing_enabled: bool
    domain_routing_min_confidence: float
    interpretation_enabled: bool
    deepseek_timeout_seconds: int
    case_workflow_enabled: bool
    owner_user_id: str
    owner_name: str
    case_max_messages: int
    case_card_template_id: str

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> "BotConfig":
        required = (
            "DEEPSEEK_API_KEY",
            "DINGTALK_CLIENT_ID",
            "DINGTALK_CLIENT_SECRET",
            "DINGTALK_ROBOT_CODE",
            "DWS_PROFILE",
            "MUMU_OPERATOR_USER_ID",
            "MUMU_ALLOWED_USER_IDS",
        )
        missing = [name for name in required if not environ.get(name, "").strip()]
        if missing:
            raise ConfigError(f"缺少运行配置: {', '.join(missing)}")

        model = environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()
        if model != "deepseek-v4-flash":
            raise ConfigError("DEEPSEEK_MODEL 必须为 deepseek-v4-flash")

        base_url = environ.get(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com",
        ).strip().rstrip("/")
        if base_url != "https://api.deepseek.com":
            raise ConfigError("DEEPSEEK_BASE_URL 必须为 https://api.deepseek.com")

        skill_version = environ.get("MUMU_SKILL_VERSION", "v2").strip().lower()
        if skill_version not in {"v1", "v2"}:
            raise ConfigError("MUMU_SKILL_VERSION 必须为 v1 或 v2")

        allowed_user_ids = tuple(
            dict.fromkeys(
                user_id.strip()
                for user_id in environ["MUMU_ALLOWED_USER_IDS"].split(",")
                if user_id.strip()
            )
        )
        operator_user_id = environ["MUMU_OPERATOR_USER_ID"].strip()
        if operator_user_id not in allowed_user_ids:
            raise ConfigError("MUMU_ALLOWED_USER_IDS 必须包含人工负责人")

        retrieval_mode = environ.get("MUMU_RETRIEVAL_MODE", "legacy").strip()
        if retrieval_mode not in {"legacy", "full_table", "v2_live"}:
            raise ConfigError("MUMU_RETRIEVAL_MODE 必须为 legacy、full_table 或 v2_live")

        knowledge_refresh_seconds = cls._positive_int(
            environ,
            "MUMU_KNOWLEDGE_REFRESH_SECONDS",
            300,
        )
        full_scan_batch_size = cls._positive_int(
            environ,
            "MUMU_FULL_SCAN_BATCH_SIZE",
            40,
        )
        domain_routing_enabled = cls._boolean(
            environ,
            "MUMU_DOMAIN_ROUTING_ENABLED",
            False,
        )
        domain_routing_min_confidence = cls._bounded_float(
            environ,
            "MUMU_DOMAIN_ROUTING_MIN_CONFIDENCE",
            0.85,
            minimum=0.0,
            maximum=1.0,
        )
        interpretation_enabled = cls._boolean(
            environ,
            "MUMU_INTERPRETATION_ENABLED",
            False,
        )
        deepseek_timeout_seconds = cls._positive_int(
            environ,
            "DEEPSEEK_TIMEOUT_SECONDS",
            45,
        )
        case_workflow_enabled = cls._boolean(
            environ,
            "MUMU_CASE_WORKFLOW_ENABLED",
            False,
        )
        owner_user_id = environ.get("MUMU_OWNER_USER_ID", "").strip()
        owner_name = environ.get("MUMU_OWNER_NAME", "").strip()
        if case_workflow_enabled:
            missing_case_config = [
                name
                for name, value in (
                    ("MUMU_OWNER_USER_ID", owner_user_id),
                    ("MUMU_OWNER_NAME", owner_name),
                )
                if not value
            ]
            if missing_case_config:
                raise ConfigError(
                    f"启用案件流程时缺少配置: {', '.join(missing_case_config)}"
                )
        case_max_messages = cls._bounded_int(
            environ,
            "MUMU_CASE_MAX_MESSAGES",
            500,
            maximum=5000,
        )
        case_card_template_id = environ.get(
            "MUMU_CASE_CARD_TEMPLATE_ID",
            "replace-with-local-template-id.schema",
        ).strip()
        if not case_card_template_id:
            raise ConfigError("MUMU_CASE_CARD_TEMPLATE_ID 不能为空")
        store_guide_workspace_id = environ.get(
            "MUMU_STORE_GUIDE_WORKSPACE_ID",
            "replace-with-local-workspace-id",
        ).strip()
        if not store_guide_workspace_id:
            raise ConfigError("MUMU_STORE_GUIDE_WORKSPACE_ID 不能为空")

        return cls(
            deepseek_api_key=environ["DEEPSEEK_API_KEY"].strip(),
            deepseek_base_url=base_url,
            deepseek_model=model,
            skill_version=skill_version,
            dingtalk_client_id=environ["DINGTALK_CLIENT_ID"].strip(),
            dingtalk_client_secret=environ["DINGTALK_CLIENT_SECRET"].strip(),
            dingtalk_robot_code=environ["DINGTALK_ROBOT_CODE"].strip(),
            dws_profile=environ["DWS_PROFILE"].strip(),
            store_guide_workspace_id=store_guide_workspace_id,
            operator_user_id=operator_user_id,
            allowed_user_ids=allowed_user_ids,
            retrieval_mode=retrieval_mode,
            knowledge_refresh_seconds=knowledge_refresh_seconds,
            full_scan_batch_size=full_scan_batch_size,
            domain_routing_enabled=domain_routing_enabled,
            domain_routing_min_confidence=domain_routing_min_confidence,
            interpretation_enabled=interpretation_enabled,
            deepseek_timeout_seconds=deepseek_timeout_seconds,
            case_workflow_enabled=case_workflow_enabled,
            owner_user_id=owner_user_id,
            owner_name=owner_name,
            case_max_messages=case_max_messages,
            case_card_template_id=case_card_template_id,
        )

    @staticmethod
    def _boolean(
        environ: Mapping[str, str],
        name: str,
        default: bool,
    ) -> bool:
        raw_value = environ.get(name, str(default)).strip().lower()
        if raw_value == "true":
            return True
        if raw_value == "false":
            return False
        raise ConfigError(f"{name} 必须为 true 或 false")

    @classmethod
    def _bounded_int(
        cls,
        environ: Mapping[str, str],
        name: str,
        default: int,
        *,
        maximum: int,
    ) -> int:
        value = cls._positive_int(environ, name, default)
        if value > maximum:
            raise ConfigError(f"{name} 必须小于等于 {maximum}")
        return value

    @staticmethod
    def _positive_int(
        environ: Mapping[str, str],
        name: str,
        default: int,
    ) -> int:
        raw_value = environ.get(name, str(default)).strip()
        try:
            value = int(raw_value)
        except ValueError as error:
            raise ConfigError(f"{name} 必须为正整数") from error
        if value < 1:
            raise ConfigError(f"{name} 必须为正整数")
        return value

    @staticmethod
    def _bounded_float(
        environ: Mapping[str, str],
        name: str,
        default: float,
        *,
        minimum: float,
        maximum: float,
    ) -> float:
        raw_value = environ.get(name, str(default)).strip()
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ConfigError(f"{name} 必须为 {minimum} 到 {maximum} 之间的数字") from error
        if not minimum <= value <= maximum:
            raise ConfigError(f"{name} 必须为 {minimum} 到 {maximum} 之间的数字")
        return value
