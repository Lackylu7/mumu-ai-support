from __future__ import annotations

import json
import asyncio
import logging
from datetime import datetime
from threading import Lock
from collections.abc import Callable
from typing import Protocol

import requests
from dingtalk_stream import CallbackHandler, CallbackMessage, CardCallbackMessage
from dingtalk_stream.utils import DINGTALK_OPENAPI_ENDPOINT


CASE_CARD_PREFIX = "mumu-case-"
CASE_RESOLVE_ACTION_ID = "mumu_case_resolve"
DEFAULT_CASE_CARD_TEMPLATE_ID = "replace-with-local-template-id.schema"


logger = logging.getLogger(__name__)


class CaseCardError(RuntimeError):
    pass


class HttpResponse(Protocol):
    def raise_for_status(self) -> None: ...


HttpRequest = Callable[..., HttpResponse]


def _http_error_details(error: Exception) -> tuple[object, str, str, str]:
    response = getattr(error, "response", None)
    if response is None:
        return "unknown", "", str(error)[:300], ""
    status = getattr(response, "status_code", "unknown")
    payload: object = None
    try:
        payload = response.json()
    except Exception:
        payload = None
    if not isinstance(payload, dict):
        text = getattr(response, "text", "")
        return status, "", str(text)[:300], ""
    code = payload.get("code", payload.get("errcode", ""))
    message = payload.get("message", payload.get("errmsg", ""))
    request_id = payload.get("requestId", payload.get("request_id", ""))
    return status, str(code)[:100], str(message)[:300], str(request_id)[:100]


def _raise_card_http_error(action: str, error: Exception) -> None:
    status, code, message, request_id = _http_error_details(error)
    logger.warning(
        "case_card_http_failed action=%s status=%s code=%s message=%s request_id=%s",
        action,
        status,
        code,
        message,
        request_id,
    )
    details = f"status={status}"
    if code:
        details += f"; code={code}"
    if message:
        details += f"; message={message}"
    raise CaseCardError(f"互动卡片{action}失败（{details}）") from error


def case_card_instance_id(case_id: str) -> str:
    normalized = case_id.strip()
    if not normalized:
        raise ValueError("案件编号不能为空")
    return f"{CASE_CARD_PREFIX}{normalized}"


def case_id_from_card_instance(card_instance_id: str) -> str:
    if not card_instance_id.startswith(CASE_CARD_PREFIX):
        return ""
    return card_instance_id[len(CASE_CARD_PREFIX) :].strip()


def _buttons(enabled: bool) -> list[dict[str, object]]:
    if not enabled:
        return []
    return [
        {
            "text": "已解决",
            "color": "blue",
            "id": CASE_RESOLVE_ACTION_ID,
            "request": True,
        }
    ]


class CaseCardGateway:
    def __init__(
        self,
        *,
        access_token_provider: Callable[[], str],
        robot_code: str,
        card_template_id: str = DEFAULT_CASE_CARD_TEMPLATE_ID,
        http_post: HttpRequest = requests.post,
        http_put: HttpRequest = requests.put,
        timeout: float = 10.0,
    ) -> None:
        if not robot_code.strip():
            raise ValueError("robotCode 不能为空")
        if not card_template_id.strip():
            raise ValueError("互动卡片模板 ID 不能为空")
        self.access_token_provider = access_token_provider
        self.robot_code = robot_code.strip()
        self.card_template_id = card_template_id.strip()
        self.http_post = http_post
        self.http_put = http_put
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        access_token = self.access_token_provider()
        if not access_token:
            raise CaseCardError("无法获取钉钉 access token")
        return {
            "Content-Type": "application/json",
            "Accept": "*/*",
            "x-acs-dingtalk-access-token": access_token,
        }

    @staticmethod
    def _card_data(
        markdown: str,
        *,
        state: str,
        button_enabled: bool,
    ) -> dict[str, str]:
        return {
            "msgTitle": "木木业务答疑案件",
            "staticMsgContent": markdown,
            "caseState": state,
            "flowStatus": "3",
            "sys_full_json_obj": json.dumps(
                {
                    "order": [
                        "msgTitle",
                        "staticMsgContent",
                        "msgButtons",
                    ],
                    "msgButtons": _buttons(button_enabled),
                },
                ensure_ascii=False,
            ),
        }

    def send(
        self,
        group_id: str,
        case_id: str,
        markdown: str,
        at_user_ids: tuple[str, ...] = (),
    ) -> str:
        if not group_id.strip():
            raise ValueError("案件群 ID 不能为空")
        instance_id = case_card_instance_id(case_id)
        body: dict[str, object] = {
            "cardTemplateId": self.card_template_id,
            "outTrackId": instance_id,
            "cardData": {
                "cardParamMap": self._card_data(
                    markdown,
                    state="处理中",
                    button_enabled=True,
                )
            },
            "callbackType": "STREAM",
            "openSpaceId": f"dtv1.card//IM_GROUP.{group_id.strip()}",
            "imGroupOpenSpaceModel": {"supportForward": False},
            "imGroupOpenDeliverModel": {
                "robotCode": self.robot_code,
                "atUserIds": {
                    user_id: user_id
                    for user_id in dict.fromkeys(at_user_ids)
                    if user_id
                },
            },
        }
        try:
            response = self.http_post(
                f"{DINGTALK_OPENAPI_ENDPOINT}/v1.0/card/instances/createAndDeliver",
                headers=self._headers(),
                json=body,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except CaseCardError:
            raise
        except Exception as error:
            _raise_card_http_error("发送", error)
        return instance_id

    def update(
        self,
        case_id: str,
        markdown: str,
        *,
        state: str,
        button_enabled: bool,
    ) -> None:
        body: dict[str, object] = {
            "outTrackId": case_card_instance_id(case_id),
            "cardData": {
                "cardParamMap": self._card_data(
                    markdown,
                    state=state,
                    button_enabled=button_enabled,
                )
            },
        }
        try:
            response = self.http_put(
                f"{DINGTALK_OPENAPI_ENDPOINT}/v1.0/card/instances",
                headers=self._headers(),
                json=body,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except CaseCardError:
            raise
        except Exception as error:
            _raise_card_http_error("更新", error)


def _action_id(content: object) -> str:
    if not isinstance(content, dict):
        return ""
    for name in ("actionId", "action_id"):
        value = content.get(name)
        if isinstance(value, str) and value:
            return value
    for nested_name in ("cardPrivateData", "params"):
        nested = content.get(nested_name)
        if isinstance(nested, str):
            try:
                nested = json.loads(nested)
            except (TypeError, ValueError):
                nested = None
        nested_action_id = _action_id(nested)
        if nested_action_id:
            return nested_action_id
    action_ids = content.get("actionIds")
    if isinstance(action_ids, list):
        for value in action_ids:
            if isinstance(value, str) and value:
                return value
    value = content.get("id")
    if isinstance(value, str) and value:
        return value
    return ""


def _callback_debug_shape(value: object, *, depth: int = 0) -> object:
    if depth >= 4:
        return f"<{type(value).__name__}>"
    if isinstance(value, dict):
        return {
            str(key): _callback_debug_shape(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _callback_debug_shape(item, depth=depth + 1)
            for item in value[:10]
        ]
    if isinstance(value, str):
        if value == CASE_RESOLVE_ACTION_ID or "resolve" in value.lower():
            return value[:100]
        return f"<str:{len(value)}>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"<{type(value).__name__}>"


class MumuCaseCardHandler(CallbackHandler):
    def __init__(
        self,
        *,
        resolution: object,
        card_gateway: object,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__()
        self.resolution = resolution
        self.card_gateway = card_gateway
        self.clock = clock or (lambda: datetime.now().astimezone())
        self._state_lock = Lock()
        self._in_progress: set[str] = set()

    async def process(self, callback: CallbackMessage) -> tuple[int, str]:
        try:
            incoming = CardCallbackMessage.from_dict(callback.data)
        except Exception as error:
            logger.warning(
                "case_card_callback_parse_failed data_keys=%s error_type=%s",
                sorted(callback.data) if isinstance(callback.data, dict) else [],
                type(error).__name__,
            )
            return 200, "OK"
        action_id = _action_id(incoming.content)
        logger.warning(
            "case_card_callback_received card_instance_id=%s action_id=%s content_keys=%s content_shape=%s",
            incoming.card_instance_id,
            action_id,
            sorted(incoming.content) if isinstance(incoming.content, dict) else [],
            json.dumps(
                _callback_debug_shape(incoming.content),
                ensure_ascii=True,
                separators=(",", ":"),
            ),
        )
        if action_id != CASE_RESOLVE_ACTION_ID:
            return 200, "OK"
        case_id = case_id_from_card_instance(incoming.card_instance_id)
        if not case_id:
            return 200, "OK"
        with self._state_lock:
            if case_id in self._in_progress:
                return 200, "OK"
            self._in_progress.add(case_id)
        extension = incoming.extension if isinstance(incoming.extension, dict) else {}
        trigger_name = extension.get("userName")
        if not isinstance(trigger_name, str) or not trigger_name.strip():
            trigger_name = incoming.user_id
        try:
            await asyncio.to_thread(
                self.card_gateway.update,
                case_id,
                "已点击，正在分析群聊并整理知识，请稍候。",
                state="沉淀中",
                button_enabled=False,
            )
        except Exception as error:
            logger.warning(
                "case_card_update_failed case_id=%s state=processing error_type=%s",
                case_id,
                type(error).__name__,
            )
        asyncio.create_task(
            self._resolve_in_background(
                case_id,
                incoming.user_id,
                trigger_name,
            )
        )
        return 200, "OK"

    async def _resolve_in_background(
        self,
        case_id: str,
        trigger_user_id: str,
        trigger_name: str,
    ) -> None:
        try:
            result = await asyncio.to_thread(
                self.resolution.resolve,
                case_id,
                trigger_user_id,
                trigger_name,
                self.clock(),
                trusted_answer=True,
            )
            if result.state == "draft":
                state = "已生成草稿"
                markdown = (
                    "群聊信息仍不完整或存在冲突，已生成不可检索的知识草稿。\n\n"
                    f"知识编号：{result.knowledge_id}"
                )
            else:
                state = "已发布"
                markdown = (
                    "本次人工答疑已经整理并发布为正式知识。\n\n"
                    f"知识编号：{result.knowledge_id}"
                )
            await asyncio.to_thread(
                self.card_gateway.update,
                case_id,
                markdown,
                state=state,
                button_enabled=False,
            )
        except Exception as error:
            logger.warning(
                "case_card_resolution_failed case_id=%s error_type=%s",
                case_id,
                type(error).__name__,
            )
            try:
                await asyncio.to_thread(
                    self.card_gateway.update,
                    case_id,
                    "已点击。本次知识沉淀暂时失败，群聊内容已保留；请在知识缺口中补充负责人答复并提交沉淀。",
                    state="沉淀失败",
                    button_enabled=False,
                )
            except Exception as update_error:
                logger.warning(
                    "case_card_update_failed case_id=%s state=failed error_type=%s",
                    case_id,
                    type(update_error).__name__,
                )
        finally:
            with self._state_lock:
                self._in_progress.discard(case_id)
