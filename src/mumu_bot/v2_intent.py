from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class V2IntentRoute:
    code: str
    signals: tuple[str, ...]
    topics: tuple[str, ...]
    domains: tuple[str, ...]


V2_INTENT_ROUTES: tuple[V2IntentRoute, ...] = (
    V2IntentRoute("PAY", ("工资", "发薪", "课时费", "提成", "绩效", "报销", "考勤", "请假", "宿舍"), ("薪资与发放", "薪酬与课时费", "绩效方案与版本", "考勤与请假", "差旅与报销", "外训与差旅", "宿舍与住宿"), ("人力与行政", "公司通用")),
    V2IntentRoute("DIV", ("分红", "风险金", "退股", "损益表", "负债", "收入数据"), ("分红与收入", "风险金与退出结算", "报表与经营数据"), ("内部政策示例", "员工合伙制", "公司通用")),
    V2IntentRoute("PROMO", ("晋升", "入股", "股东", "开店", "族长", "同进同退", "竞业", "竞聘"), ("股东晋升与资格", "开店与门店发展", "合伙制度说明", "股东管理与门店权限", "行为红线与合规"), ("内部政策示例", "员工合伙制", "公司通用")),
    V2IntentRoute("ADMIN", ("OA", "用印", "离职手续", "转岗", "行政", "培训教室"), ("OA与用印", "入转调离", "培训与行政支持"), ("人力与行政", "教练运营", "公司通用")),
    V2IntentRoute("HIRE", ("招聘", "面试", "入职", "转正", "报到"), ("招聘与招募", "入职与上岗", "转正与评审"), ("人力与行政", "公司通用")),
    V2IntentRoute("TRAIN", ("考核", "培训", "分级", "认证", "师徒", "学习"), ("培训与技术考核", "培训与上岗", "晋升与分级", "学习与成长"), ("教练运营", "人力与行政", "公司通用")),
    V2IntentRoute("STORE", ("物资", "采购", "盘点", "闭店", "门店环境", "管理绩效"), ("物资与采购", "闭店与营业保障", "门店环境与设施", "管理与绩效", "教练全流程与数据"), ("门店运营", "教练运营", "公司通用")),
    V2IntentRoute("LOCATE", ("门店", "店在哪", "地址", "床位", "几点开", "营业"), (), ("公司通用",)),
    V2IntentRoute("CS", ("疼", "痛", "麻", "晕", "难受", "差评", "退款", "会员", "服务礼仪"), ("服务安全与风险", "客户服务与安全", "会员服务", "客户售后与争议", "服务礼仪与品牌"), ("客户服务与品牌", "公司通用")),
    V2IntentRoute("MEET", ("会议", "分享", "主持人", "KING奖"), ("会议与分享",), ("公司通用",)),
    V2IntentRoute("BOT", ("木木", "机器人", "重试", "反馈"), ("机器人使用与反馈",), ("公司通用",)),
)


def v2_intent_routes(question: str) -> tuple[V2IntentRoute, ...]:
    normalized = re.sub(r"\s+", "", question).lower()
    return tuple(
        route
        for route in V2_INTENT_ROUTES
        if any(signal.lower() in normalized for signal in route.signals)
    )


def v2_topic_values(question: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(topic for route in v2_intent_routes(question) for topic in route.topics))


def v2_domain_values(question: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(domain for route in v2_intent_routes(question) for domain in route.domains))


def v2_keyword_variants(question: str) -> tuple[str, ...]:
    normalized = re.sub(r"[，。！？!?、；;：:（）()【】\[\]\"'“”‘’]+", " ", question)
    variants = [" ".join(normalized.split())]
    for route in v2_intent_routes(question):
        variants.extend(signal for signal in route.signals if signal in question)
    return tuple(item for item in dict.fromkeys(variants) if item)[:4]

