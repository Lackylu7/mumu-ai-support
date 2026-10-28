import json
import unittest
import asyncio
from datetime import datetime, timezone

from dingtalk_stream import CallbackMessage

from mumu_bot.case_card import (
    CASE_RESOLVE_ACTION_ID,
    CaseCardError,
    CaseCardGateway,
    MumuCaseCardHandler,
    case_card_instance_id,
    case_id_from_card_instance,
)


class FakeResponse:
    def __init__(self) -> None:
        self.raised = False

    def raise_for_status(self) -> None:
        self.raised = True


class FakeRequest:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str], dict[str, object], float]] = []

    def __call__(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> FakeResponse:
        self.calls.append((url, headers, json, timeout))
        return FakeResponse()


class FakeErrorResponse:
    status_code = 400

    def json(self):
        return {
            "code": "InvalidParameter",
            "message": "card template is invalid",
            "requestId": "request-123",
        }


class FakeHttpError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("400 Client Error")
        self.response = FakeErrorResponse()


class FakeFailedResponse:
    def raise_for_status(self) -> None:
        raise FakeHttpError()


def failed_request(*args, **kwargs):
    del args, kwargs
    return FakeFailedResponse()


class CaseCardGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.post = FakeRequest()
        self.put = FakeRequest()
        self.gateway = CaseCardGateway(
            access_token_provider=lambda: "access-token",
            robot_code="robot-code",
            card_template_id="template.schema",
            http_post=self.post,
            http_put=self.put,
            timeout=8.0,
        )
        self.case_id = "MM-GAP-20260811-ABC123"

    def test_card_instance_id_round_trips_case_id(self) -> None:
        instance_id = case_card_instance_id(self.case_id)

        self.assertEqual(instance_id, "mumu-case-MM-GAP-20260811-ABC123")
        self.assertEqual(case_id_from_card_instance(instance_id), self.case_id)
        self.assertEqual(case_id_from_card_instance("unrelated-card"), "")

    def test_send_delivers_one_request_button_to_case_group(self) -> None:
        instance_id = self.gateway.send(
            "cid-case",
            self.case_id,
            "案件说明正文",
        )

        self.assertEqual(instance_id, case_card_instance_id(self.case_id))
        self.assertEqual(len(self.post.calls), 1)
        url, headers, body, timeout = self.post.calls[0]
        self.assertTrue(url.endswith("/v1.0/card/instances/createAndDeliver"))
        self.assertEqual(headers["x-acs-dingtalk-access-token"], "access-token")
        self.assertEqual(timeout, 8.0)
        self.assertEqual(body["openSpaceId"], "dtv1.card//IM_GROUP.cid-case")
        self.assertEqual(body["outTrackId"], case_card_instance_id(self.case_id))
        self.assertEqual(body["callbackType"], "STREAM")
        card_map = body["cardData"]["cardParamMap"]
        self.assertEqual(card_map["staticMsgContent"], "案件说明正文")
        full_card = json.loads(card_map["sys_full_json_obj"])
        self.assertEqual(
            full_card["order"],
            ["msgTitle", "staticMsgContent", "msgButtons"],
        )
        buttons = full_card["msgButtons"]
        self.assertEqual(
            buttons,
            [
                {
                    "text": "已解决",
                    "color": "blue",
                    "id": CASE_RESOLVE_ACTION_ID,
                    "request": True,
                }
            ],
        )

    def test_update_final_state_removes_button(self) -> None:
        self.gateway.update(
            self.case_id,
            "知识已经发布",
            state="已发布",
            button_enabled=False,
        )

        self.assertEqual(len(self.put.calls), 1)
        url, _, body, _ = self.put.calls[0]
        self.assertTrue(url.endswith("/v1.0/card/instances"))
        self.assertEqual(body["outTrackId"], case_card_instance_id(self.case_id))
        card_map = body["cardData"]["cardParamMap"]
        self.assertEqual(card_map["caseState"], "已发布")
        self.assertEqual(json.loads(card_map["sys_full_json_obj"])["msgButtons"], [])

    def test_send_keeps_safe_dingtalk_error_details(self) -> None:
        gateway = CaseCardGateway(
            access_token_provider=lambda: "access-token",
            robot_code="robot-code",
            card_template_id="template.schema",
            http_post=failed_request,
        )

        with self.assertRaisesRegex(
            CaseCardError,
            "status=400; code=InvalidParameter; message=card template is invalid",
        ):
            gateway.send("cid-case", self.case_id, "案件说明正文")


class FakeResolution:
    def __init__(self, state: str = "published") -> None:
        self.state = state
        self.calls: list[tuple[str, str, str]] = []

    def resolve(
        self,
        case_id,
        trigger_user_id,
        trigger_name,
        now,
        *,
        trusted_answer=False,
    ):
        del now
        self.calls.append(
            (case_id, trigger_user_id, trigger_name, trusted_answer)
        )
        return type(
            "Result",
            (),
            {
                "state": self.state,
                "message": "处理完成",
                "knowledge_id": "MUMU-AUTO-ABC",
            },
        )()


class FakeCardUpdater:
    def __init__(self) -> None:
        self.updates: list[tuple[str, str, str, bool]] = []

    def update(self, case_id, markdown, *, state, button_enabled):
        self.updates.append((case_id, markdown, state, button_enabled))


def card_callback(case_id: str, action_id: str = CASE_RESOLVE_ACTION_ID):
    callback = CallbackMessage()
    callback.data = {
        "outTrackId": case_card_instance_id(case_id),
        "userId": "member-1",
        "extension": json.dumps({"userName": "群成员"}, ensure_ascii=False),
        "content": json.dumps({"actionId": action_id}, ensure_ascii=False),
    }
    return callback


def real_card_callback(case_id: str, action_id: str = CASE_RESOLVE_ACTION_ID):
    callback = CallbackMessage()
    callback.data = {
        "outTrackId": case_card_instance_id(case_id),
        "userId": "member-1",
        "extension": json.dumps({"userName": "群成员"}, ensure_ascii=False),
        "content": json.dumps(
            {"cardPrivateData": {"actionId": action_id}},
            ensure_ascii=False,
        ),
    }
    return callback


def actual_dingtalk_card_callback(
    case_id: str,
    action_id: str = CASE_RESOLVE_ACTION_ID,
):
    callback = CallbackMessage()
    callback.data = {
        "outTrackId": case_card_instance_id(case_id),
        "userId": "member-1",
        "extension": json.dumps({"userName": "群成员"}, ensure_ascii=False),
        "content": json.dumps(
            {
                "cardPrivateData": {
                    "actionIds": ["single_button_node_ocljy2j7wg2"],
                    "params": {"id": action_id, "text": "已解决"},
                }
            },
            ensure_ascii=False,
        ),
    }
    return callback


class CaseCardHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_dingtalk_action_ids_callback_locks_card(self) -> None:
        resolution = FakeResolution()
        cards = FakeCardUpdater()
        handler = MumuCaseCardHandler(
            resolution=resolution,
            card_gateway=cards,
        )

        result = await handler.process(
            actual_dingtalk_card_callback("MM-GAP-20260811-ACTUAL")
        )

        self.assertEqual(result, (200, "OK"))
        self.assertEqual(len(cards.updates), 1)
        self.assertEqual(cards.updates[0][2:], ("沉淀中", False))

    async def test_real_nested_card_action_starts_resolution(self) -> None:
        resolution = FakeResolution()
        cards = FakeCardUpdater()
        handler = MumuCaseCardHandler(
            resolution=resolution,
            card_gateway=cards,
        )

        result = await handler.process(
            real_card_callback("MM-GAP-20260811-NESTED")
        )
        await asyncio.sleep(0.05)

        self.assertEqual(result, (200, "OK"))
        self.assertEqual(len(resolution.calls), 1)

    async def test_card_click_treats_group_conclusion_as_trusted(self) -> None:
        resolution = FakeResolution()
        cards = FakeCardUpdater()
        handler = MumuCaseCardHandler(
            resolution=resolution,
            card_gateway=cards,
        )

        await handler.process(
            actual_dingtalk_card_callback("MM-GAP-20260813-TRUSTED")
        )
        await asyncio.sleep(0.05)

        self.assertEqual(
            resolution.calls,
            [
                (
                    "MM-GAP-20260813-TRUSTED",
                    "member-1",
                    "群成员",
                    True,
                )
            ],
        )

    async def test_first_click_locks_card_before_callback_is_acked(self) -> None:
        resolution = FakeResolution()
        cards = FakeCardUpdater()
        handler = MumuCaseCardHandler(
            resolution=resolution,
            card_gateway=cards,
        )

        result = await handler.process(
            real_card_callback("MM-GAP-20260811-LOCK")
        )

        self.assertEqual(result, (200, "OK"))
        self.assertEqual(len(cards.updates), 1)
        self.assertIn("已点击", cards.updates[0][1])
        self.assertEqual(cards.updates[0][2:], ("沉淀中", False))

    async def test_failed_resolution_keeps_card_button_locked(self) -> None:
        class FailedResolution:
            def resolve(self, *args):
                del args
                raise RuntimeError("resolution failed")

        cards = FakeCardUpdater()
        handler = MumuCaseCardHandler(
            resolution=FailedResolution(),
            card_gateway=cards,
        )

        await handler.process(real_card_callback("MM-GAP-20260811-FAIL"))
        await asyncio.sleep(0.05)

        self.assertEqual(cards.updates[-1][2:], ("沉淀失败", False))

    async def test_resolve_button_acks_then_updates_card(self) -> None:
        resolution = FakeResolution()
        cards = FakeCardUpdater()
        handler = MumuCaseCardHandler(
            resolution=resolution,
            card_gateway=cards,
            clock=lambda: datetime(2026, 8, 11, tzinfo=timezone.utc),
        )

        result = await handler.process(card_callback("MM-GAP-20260811-ABC"))
        await asyncio.sleep(0.05)

        self.assertEqual(result, (200, "OK"))
        self.assertEqual(
            resolution.calls,
            [("MM-GAP-20260811-ABC", "member-1", "群成员", True)],
        )
        self.assertEqual(cards.updates[0][2:], ("沉淀中", False))
        self.assertEqual(cards.updates[-1][2:], ("已发布", False))

    async def test_unknown_action_is_silently_acked(self) -> None:
        resolution = FakeResolution()
        cards = FakeCardUpdater()
        handler = MumuCaseCardHandler(
            resolution=resolution,
            card_gateway=cards,
        )

        result = await handler.process(
            card_callback("MM-GAP-20260811-ABC", action_id="unknown")
        )
        await asyncio.sleep(0)

        self.assertEqual(result, (200, "OK"))
        self.assertEqual(resolution.calls, [])
        self.assertEqual(cards.updates, [])


if __name__ == "__main__":
    unittest.main()
