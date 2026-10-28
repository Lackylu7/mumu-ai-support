from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .case_domain import CaseMessage


class CaseGatewayError(RuntimeError):
    pass


class CaseMessageLimitExceeded(CaseGatewayError):
    pass


def _data(payload: object) -> object:
    value = payload
    for _ in range(4):
        if not isinstance(value, dict):
            break
        if "data" in value:
            value = value["data"]
            continue
        if "result" in value:
            value = value["result"]
            continue
        break
    return value


def _string(value: object, *names: str) -> str:
    if not isinstance(value, dict):
        return ""
    for name in names:
        candidate = value.get(name)
        if isinstance(candidate, str) and candidate:
            return candidate
    return ""


def _items(payload: object, *names: str) -> list[dict[str, object]]:
    value = _data(payload)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for name in names:
            candidate = value.get(name)
            if isinstance(candidate, list):
                return [item for item in candidate if isinstance(item, dict)]
    return []


def _message_text(item: dict[str, object]) -> str:
    direct = _string(item, "content", "messageContent")
    if direct:
        return direct
    for name in ("text", "content"):
        value = item.get(name)
        if isinstance(value, dict):
            nested = _string(value, "content", "text")
            if nested:
                return nested
    return ""


def _created_at(item: dict[str, object]) -> datetime | None:
    value = item.get("createTime", item.get("createdAt"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1000
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
            return parsed
        except ValueError:
            return None
    return None


class DwsCaseGateway:
    def __init__(
        self,
        client: Any,
        *,
        robot_code: str,
        card_gateway: Any | None = None,
    ) -> None:
        if not robot_code.strip():
            raise ValueError("robotCode 不能为空")
        self.client = client
        self.robot_code = robot_code
        self.card_gateway = card_gateway

    def send_case_card(
        self,
        group_id: str,
        case_id: str,
        markdown: str,
        at_user_ids: tuple[str, ...] = (),
    ) -> str:
        if self.card_gateway is None:
            raise CaseGatewayError("互动卡片网关未配置")
        return self.card_gateway.send(group_id, case_id, markdown, at_user_ids)

    def update_case_card(
        self,
        case_id: str,
        markdown: str,
        *,
        state: str,
        button_enabled: bool,
    ) -> None:
        if self.card_gateway is None:
            raise CaseGatewayError("互动卡片网关未配置")
        self.card_gateway.update(
            case_id,
            markdown,
            state=state,
            button_enabled=button_enabled,
        )

    def create_group(self, name: str, member_ids: tuple[str, ...]) -> str:
        unique_members = tuple(
            dict.fromkeys(item.strip() for item in member_ids if item.strip())
        )
        if not unique_members:
            raise CaseGatewayError("建群成员不能为空")
        payload = self.client.create_chat_group(name, ",".join(unique_members))
        group_id = _string(
            _data(payload),
            "openConversationId",
            "openconversation_id",
            "conversationId",
            "id",
        )
        if not group_id:
            raise CaseGatewayError("创建群后未返回群会话 ID")
        return group_id

    def find_group_by_exact_name(self, name: str) -> str:
        cursor = "0"
        while True:
            payload = self.client.search_chat_groups(name, limit=20, cursor=cursor)
            for item in _items(payload, "items", "groups", "records"):
                title = _string(item, "title", "name", "groupName")
                if title == name:
                    return _string(
                        item,
                        "openConversationId",
                        "openconversation_id",
                        "conversationId",
                        "id",
                    )
            data = _data(payload)
            if not isinstance(data, dict) or not data.get("hasMore"):
                return ""
            next_cursor = _string(data, "nextCursor", "cursor")
            if not next_cursor or next_cursor == cursor:
                return ""
            cursor = next_cursor

    def add_bot(self, group_id: str) -> None:
        try:
            self.client.add_bot_to_group(group_id, self.robot_code)
        except Exception:
            if self.bot_is_present(group_id):
                return
            raise

    def bot_is_present(self, group_id: str) -> bool:
        payload = self.client.list_group_bots(group_id)
        return any(
            _string(item, "robotCode", "robot_code") == self.robot_code
            for item in _items(payload, "bots", "items", "records")
        )

    def send_bot_message(
        self,
        group_id: str,
        title: str,
        text: str,
        at_user_ids: tuple[str, ...] = (),
    ) -> str:
        payload = self.client.send_chat_message_by_bot(
            robot_code=self.robot_code,
            group_id=group_id,
            title=title,
            text=text,
            at_user_ids=tuple(dict.fromkeys(at_user_ids)),
        )
        return _string(
            _data(payload),
            "processQueryKey",
            "messageId",
            "msgId",
            "id",
        )

    def list_messages(
        self,
        group_id: str,
        since: datetime,
        *,
        limit: int,
    ) -> tuple[CaseMessage, ...]:
        if limit < 1 or limit > 500:
            raise ValueError("消息上限必须在 1 到 500 之间")
        boundary = since
        collected: list[CaseMessage] = []
        seen_ids: set[str] = set()
        while True:
            collected_before_page = len(collected)
            remaining = limit - len(collected)
            payload = self.client.list_chat_messages(
                group_id=group_id,
                start_time=boundary.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                direction="newer",
                limit=max(1, min(100, remaining)),
            )
            page_latest = boundary
            for item in _items(payload, "messages", "items", "records"):
                message_id = _string(
                    item,
                    "msgId",
                    "messageId",
                    "openMessageId",
                    "id",
                )
                created_at = _created_at(item)
                if created_at is not None and created_at > page_latest:
                    page_latest = created_at
                sender_type = _string(item, "senderType", "sender_type").lower()
                message_type = _string(item, "msgType", "messageType", "type").lower()
                sender_name = _string(item, "senderNick", "senderName", "sender")
                text = _message_text(item).strip()
                if (
                    not message_id
                    or message_id in seen_ids
                    or sender_type in {"robot", "bot", "system"}
                    or message_type in {"system", "event"}
                    or sender_name == "木木"
                    or not text
                ):
                    continue
                seen_ids.add(message_id)
                collected.append(
                    CaseMessage(
                        message_id=message_id,
                        sender_user_id=_string(
                            item,
                            "senderStaffId",
                            "senderUserId",
                            "senderOpenDingTalkId",
                            "senderId",
                        ),
                        sender_name=sender_name,
                        text=text,
                        created_at=created_at or boundary,
                    )
                )
                if len(collected) >= limit:
                    break
            data = _data(payload)
            has_more = isinstance(data, dict) and data.get("hasMore") is True
            if len(collected) >= limit and has_more:
                raise CaseMessageLimitExceeded("群消息超过安全上限")
            if not has_more:
                return tuple(sorted(collected, key=lambda item: item.created_at))
            if (
                page_latest <= boundary
                and len(collected) == collected_before_page
            ):
                raise CaseGatewayError("群消息分页边界没有推进")
            if page_latest > boundary:
                boundary = page_latest
