from __future__ import annotations

import json
from urllib.request import Request

from mumu_bot.deepseek import DeepSeekClient
from mumu_bot.domain import Evidence


class FakeTransport:
    def __init__(self) -> None:
        self.request: Request | None = None

    def send(self, request: Request, timeout: float) -> bytes:
        del timeout
        self.request = request
        return b'{"choices":[{"message":{"content":"ok"}}]}'


def _evidence() -> Evidence:
    return Evidence(
        knowledge_id="K1",
        intent="测试规则",
        answer="按正式规则办理。",
        source_id="S1",
        source_section="第一节",
        version="1.0",
        risk_level="普通",
        sensitivity="公开内部",
        business_domain="公司通用",
    )


def test_v2_uses_new_skill_without_legacy_voice_layers() -> None:
    transport = FakeTransport()
    client = DeepSeekClient(
        api_key="test",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        transport=transport,
        skill_version="v2",
    )

    client.complete("这个规则为什么这样设置？", [_evidence()])
    assert transport.request is not None
    payload = json.loads(transport.request.data.decode("utf-8"))
    prompt = payload["messages"][0]["content"]
    assert "木木知识库客服 v2" in prompt
    assert "意图分类决策树" in prompt
    assert "检索降级阶梯" in prompt
    assert "3点左右的必要信息" in prompt
    assert "不默认追加“还有其他问题吗”" in prompt
    assert "最终篇幅约束" in prompt
    assert "木木经营者视角表达" not in prompt
    assert "木木说话风格" not in prompt


def test_v1_keeps_legacy_voice_layers_for_rollback() -> None:
    transport = FakeTransport()
    client = DeepSeekClient(
        api_key="test",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        transport=transport,
        skill_version="v1",
    )

    client.complete("这个规则为什么这样设置？", [_evidence()])
    assert transport.request is not None
    payload = json.loads(transport.request.data.decode("utf-8"))
    prompt = payload["messages"][0]["content"]
    assert "木木经营者视角表达" in prompt
    assert "木木说话风格" in prompt
    assert "木木知识库客服 v2" not in prompt
