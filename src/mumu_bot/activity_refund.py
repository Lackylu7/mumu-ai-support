from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import logging
from threading import Lock
from typing import Any, Protocol

from .domain import ConversationTurn, Evidence


logger = logging.getLogger(__name__)


class RecordClient(Protocol):
    def query_all_records(self, base_id: str, table_id: str) -> dict[str, object]: ...


@dataclass(frozen=True)
class ActivityRefundRule:
    rule_id: str
    city: str
    activity: str
    card: str
    period: str
    conditions: str
    required_inputs: str
    calculation: str
    exceptions: str
    source_label: str
    source_url: str
    source_section: str


@dataclass(frozen=True)
class ActivityRefundResult:
    text: str
    evidence: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class ActivityRefundFacts:
    activity_or_card: str = ""
    card_state: str = "unknown"
    purchase_days: Decimal | None = None
    purchase_window: str = "unknown"
    paid_amount: Decimal | None = None
    remaining_balance: Decimal | None = None
    used_course_count: Decimal | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> "ActivityRefundFacts":
        if not isinstance(values, Mapping):
            return cls()
        state = values.get("card_state")
        card_state = state if state in {"unused", "used_regular", "gift_used"} else "unknown"
        purchase_window_value = values.get("purchase_window")
        purchase_window = (
            purchase_window_value
            if isinstance(purchase_window_value, str)
            and purchase_window_value in {"within_7", "over_7"}
            else "unknown"
        )
        activity = values.get("activity_or_card")
        return cls(
            activity_or_card=activity.strip() if isinstance(activity, str) else "",
            card_state=card_state,
            purchase_days=_fact_decimal(values.get("purchase_days")),
            purchase_window=purchase_window,
            paid_amount=_fact_decimal(values.get("paid_amount")),
            remaining_balance=_fact_decimal(values.get("remaining_balance")),
            used_course_count=_fact_decimal(values.get("used_course_count")),
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "activity_or_card": self.activity_or_card,
            "card_state": self.card_state,
            "purchase_days": self.purchase_days,
            "purchase_window": self.purchase_window,
            "paid_amount": self.paid_amount,
            "remaining_balance": self.remaining_balance,
            "used_course_count": self.used_course_count,
        }


@dataclass(frozen=True)
class ActivityRefundUnderstanding:
    is_activity_refund: bool
    facts: ActivityRefundFacts = ActivityRefundFacts()


def _fact_decimal(value: object) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return decimal if decimal >= 0 else None


def _cell_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("markdown", "name", "text"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip()
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _normal(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value)


def _contains_any(text: str, values: Sequence[str]) -> bool:
    return any(value and value in text for value in values)


def _question_has_refund_intent(question: str) -> bool:
    compact = _normal(question)
    return _contains_any(compact, ("退款", "退费", "退卡", "退课", "退钱"))


def _question_card_terms(rule: ActivityRefundRule) -> tuple[str, ...]:
    raw = f"{rule.activity} {rule.card}"
    parts = list(
        item
        for item in re.split(r"[+＋/、，,\s]+", raw)
        if len(_normal(item)) >= 2
    )
    # People commonly shorten “2026年6月卡项” to “6月活动卡”.  Keep the
    # semantic route AI-based, but make the deterministic rule join tolerant
    # of equivalent date/card wording.
    activity_text = _normal(rule.activity)
    for match in re.finditer(r"(?:20\d{2})?年?(\d{1,2})月", rule.activity):
        month = match.group(1)
        parts.extend((f"{month}月", f"{match.group(0)}"))
    if "活动" in activity_text:
        parts.append("活动")
    return tuple(
        dict.fromkeys(
            normalized
            for item in parts
            for normalized in (_normal(item),)
            if len(normalized) >= 2 and normalized not in {"适用该活动的卡项"}
        )
    )


def _amount_after(question: str, labels: Sequence[str]) -> Decimal | None:
    compact = unicodedata.normalize("NFKC", question).replace("，", ",")
    for label in labels:
        match = re.search(
            rf"{label}\s*(?:是|为|有|：|:|约)?\s*(\d+(?:\.\d+)?)\s*(?:元|块)?",
            compact,
        )
        if match:
            try:
                return Decimal(match.group(1))
            except InvalidOperation:
                continue
    return None


def _count_after(question: str, labels: Sequence[str]) -> Decimal | None:
    compact = unicodedata.normalize("NFKC", question)
    for label in labels:
        match = re.search(rf"{label}\s*(?:是|为|有|了|：|:)?\s*(\d+(?:\.\d+)?)\s*(?:节|次)?", compact)
        if match:
            try:
                return Decimal(match.group(1))
            except InvalidOperation:
                continue
    return None


def _elapsed_days(question: str) -> Decimal | None:
    compact = unicodedata.normalize("NFKC", question)
    patterns = (
        r"(?:购卡|开卡)[^0-9]{0,6}(\d+(?:\.\d+)?)\s*天",
        r"已经[^0-9]{0,6}(\d+(?:\.\d+)?)\s*天",
        r"过了[^0-9]{0,6}(\d+(?:\.\d+)?)\s*天",
    )
    for pattern in patterns:
        match = re.search(pattern, compact)
        if match:
            try:
                return Decimal(match.group(1))
            except InvalidOperation:
                continue
    return None


def _format_money(value: Decimal) -> str:
    rendered = format(value.quantize(Decimal("0.01")), "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _city_from_company_name(company_name: str) -> str:
    """Map only an explicit city in an employee's DingTalk organization name."""
    normalized_name = _normal(company_name)
    for city in ("深圳", "上海"):
        if _normal(city) in normalized_name:
            return city
    return ""


class ActivityRefundRepository:
    """City-gated, deterministic handling for promotion-card refunds."""

    def __init__(
        self,
        client: RecordClient,
        *,
        base_id: str,
        rule_table_id: str,
        rule_fields: Mapping[str, str],
    ) -> None:
        self.client = client
        self.base_id = base_id
        self.rule_table_id = rule_table_id
        self.rule_fields = dict(rule_fields)
        self._lock = Lock()

    def answer(
        self,
        user_id: str,
        question: str,
        history: Sequence[ConversationTurn] = (),
        company_name: str = "",
        is_activity_refund: bool | None = None,
        facts: Mapping[str, object] | None = None,
    ) -> ActivityRefundResult | None:
        # Semantic routing is the primary decision.  The keyword test remains
        # only as an availability fallback when the upstream classifier fails.
        if is_activity_refund is False:
            return None
        if is_activity_refund is None and not _question_has_refund_intent(question):
            return None

        extracted_facts = ActivityRefundFacts.from_mapping(facts)

        city = self._city_for_user(user_id, company_name)
        if not city:
            return ActivityRefundResult(
                "活动退款规则按所属公司所在城市执行，但暂时无法识别你的钉钉所属公司。"
                "请先确认公司信息后再查询。"
            )

        contextual_question = "\n".join(
            turn.question for turn in history[-3:] if turn.question.strip()
        )
        if contextual_question:
            contextual_question = f"{contextual_question}\n{question}"
        else:
            contextual_question = question
        rules = self._matching_rules(city, contextual_question, extracted_facts)
        if not rules:
            return ActivityRefundResult(
                f"已按{city}规则查询，但还无法确定具体活动或卡项。"
                "请补充卡项名称和购卡时间。"
            )

        applicable = self._applicable_rules(rules, contextual_question, extracted_facts)
        if len(applicable) != 1:
            details = "；".join(
                dict.fromkeys(rule.required_inputs for rule in rules if rule.required_inputs)
            )
            return ActivityRefundResult(
                "已定位到对应活动，但退款分支还不能唯一确定。"
                f"请补充：{details or '是否已使用、购卡时间及剩余金额'}。",
                self._evidence(rules),
            )

        rule = applicable[0]
        calculation = self._calculate(rule, contextual_question, extracted_facts)
        if calculation is None:
            calculation = rule.calculation
        text = f"按{city}《{rule.activity}》的规则：{calculation}"
        missing_inputs = self._missing_inputs_for_calculation(
            rule,
            contextual_question,
            extracted_facts,
        )
        if missing_inputs:
            text += f"\n\n还需补充：{'；'.join(missing_inputs)}。"
        if rule.exceptions:
            text += f"\n\n注意：{rule.exceptions}"
        if rule.source_label:
            source = f"木木 AI 表 > 活动退款规则 > 《{rule.source_label}》"
            if rule.source_section:
                source += f" > {rule.source_section}"
            if rule.period:
                source += f"（适用购卡期间：{rule.period}）"
            if rule.source_url.startswith(("https://", "http://")):
                source += f"\n  原文链接：{rule.source_url}"
            text += f"\n\n回答内容来源：\n- {source}"
        return ActivityRefundResult(text, self._evidence((rule,)))

    def _city_for_user(self, user_id: str, company_name: str) -> str:
        company_city = _city_from_company_name(company_name)
        if company_city:
            return company_city
        return _city_from_company_name(self._company_for_user(user_id))

    def _company_for_user(self, user_id: str) -> str:
        """Read the employee's organization name from DingTalk when available.

        The chat callback normally provides an employee ID, but not a company
        name.  DWS contact lookup returns ``orgEmployeeModel.orgName`` for
        that employee, which is the authoritative organization display name.
        The optional capability keeps unit-test clients and local tools usable.
        """
        run_json = getattr(self.client, "run_json", None)
        if not callable(run_json) or not user_id:
            return ""
        try:
            payload = run_json(["contact", "user", "get", "--ids", user_id])
        except Exception as error:
            logger.info(
                "activity_refund_company_lookup_failed user_id=%s error_type=%s",
                user_id,
                type(error).__name__,
            )
            return ""
        if not isinstance(payload, Mapping):
            return ""
        results = payload.get("result")
        if not isinstance(results, list) or len(results) != 1:
            return ""
        result = results[0]
        if not isinstance(result, Mapping):
            return ""
        employee = result.get("orgEmployeeModel")
        if not isinstance(employee, Mapping):
            return ""
        return _cell_text(employee.get("orgName"))

    @staticmethod
    def has_explicit_refund_intent(question: str) -> bool:
        return _question_has_refund_intent(question)

    def _matching_rules(
        self,
        city: str,
        question: str,
        facts: ActivityRefundFacts,
    ) -> tuple[ActivityRefundRule, ...]:
        candidates = [rule for rule in self._rules() if rule.city == city]
        normalized_question = _normal(f"{question}\n{facts.activity_or_card}")
        matched = [
            rule
            for rule in candidates
            if _contains_any(normalized_question, _question_card_terms(rule))
        ]
        return tuple(matched)

    @staticmethod
    def _applicable_rules(
        rules: Sequence[ActivityRefundRule],
        question: str,
        facts: ActivityRefundFacts,
    ) -> tuple[ActivityRefundRule, ...]:
        compact = _normal(question)
        unused = facts.card_state == "unused" or (
            facts.card_state == "unknown"
            and _contains_any(compact, ("未使用", "没使用", "没用过", "未上课", "没上课"))
        )
        used = facts.card_state in {"used_regular", "gift_used"} or (
            facts.card_state == "unknown"
            and _contains_any(compact, ("已使用", "用过", "上过", "已上课", "已用"))
        )
        gift_used = facts.card_state == "gift_used" or (
            facts.card_state == "unknown"
            and _contains_any(compact, ("赠送课已用", "用了赠课", "赠课用过", "赠送课程已使用"))
        )
        days = facts.purchase_days if facts.purchase_days is not None else _elapsed_days(question)
        normalized_question = _normal(question)
        over_seven_days = _contains_any(
            normalized_question,
            ("超过7天", "大于7天", "七天以上", "超过七天"),
        ) or facts.purchase_window == "over_7"
        within_seven_days = _contains_any(
            normalized_question,
            ("7天内", "七天内", "不超过7天", "七天以内"),
        ) or facts.purchase_window == "within_7"
        balance = (
            facts.remaining_balance
            if facts.remaining_balance is not None
            else _amount_after(question, ("剩余金额", "余额"))
        )
        result: list[ActivityRefundRule] = []
        for rule in rules:
            condition = _normal(rule.conditions)
            entirely_unused = _contains_any(
                condition,
                ("卡项未使用", "次卡未使用", "充值卡未使用"),
            )
            used_regular_without_gift = "已使用正课但未使用赠送课程" in condition
            if entirely_unused and (not unused or used):
                continue
            if used_regular_without_gift and (not used or gift_used):
                continue
            if (
                "已使用赠送课程" in condition
                and not used_regular_without_gift
                and not gift_used
            ):
                continue
            if "不超过7天" in condition and (
                over_seven_days or (days is not None and days > 7) or (days is None and not within_seven_days)
            ):
                continue
            if "超过7天" in condition and "不超过7天" not in condition and (
                within_seven_days or (days is not None and days <= 7 and not over_seven_days) or (days is None and not over_seven_days)
            ):
                continue
            if "低于400元" in condition and (balance is None or balance >= 400):
                continue
            if "400元及以上" in condition and (balance is None or balance < 400):
                continue
            result.append(rule)
        return tuple(result)

    @staticmethod
    def _calculate(
        rule: ActivityRefundRule,
        question: str,
        facts: ActivityRefundFacts,
    ) -> str | None:
        calculation = rule.calculation
        original = (
            facts.paid_amount
            if facts.paid_amount is not None
            else _amount_after(question, ("原始实付金额", "实付金额", "实付", "购买金额"))
        )
        balance = (
            facts.remaining_balance
            if facts.remaining_balance is not None
            else _amount_after(question, ("扣课后余额", "剩余金额", "余额"))
        )
        used = (
            facts.used_course_count
            if facts.used_course_count is not None
            else _count_after(question, ("已使用", "已用", "用了", "上了"))
        )
        if "- 30 元" in calculation and original is not None:
            return f"预计退款 {_format_money(max(Decimal(0), original - Decimal(30)))} 元。"
        if "× 85%" in calculation and balance is not None:
            return f"预计退款 {_format_money(balance * Decimal('0.85'))} 元。"
        if "× 479 元" in calculation and original is not None and used is not None:
            return f"预计退款 {_format_money(max(Decimal(0), original - used * Decimal(479)))} 元。"
        if "全额退款" in calculation:
            return "符合该分支时可全额退款。"
        return None

    @staticmethod
    def _missing_inputs_for_calculation(
        rule: ActivityRefundRule,
        question: str,
        facts: ActivityRefundFacts,
    ) -> tuple[str, ...]:
        calculation = rule.calculation
        original = (
            facts.paid_amount
            if facts.paid_amount is not None
            else _amount_after(
                question, ("原始实付金额", "实付金额", "实付", "购买金额")
            )
        )
        balance = (
            facts.remaining_balance
            if facts.remaining_balance is not None
            else _amount_after(question, ("扣课后余额", "剩余金额", "余额"))
        )
        used = (
            facts.used_course_count
            if facts.used_course_count is not None
            else _count_after(question, ("已使用", "已用", "用了", "上了"))
        )
        if "- 30 元" in calculation and original is None:
            return ("实付金额",)
        if "× 85%" in calculation and balance is None:
            return ("扣课后余额",)
        if "× 479 元" in calculation:
            missing: list[str] = []
            if original is None:
                missing.append("实付金额")
            if used is None:
                missing.append("已用正课节数")
            return tuple(missing)
        return ()

    def _rules(self) -> tuple[ActivityRefundRule, ...]:
        records = self._records(self.rule_table_id)
        rules: list[ActivityRefundRule] = []
        for record in records:
            cells = self._cells(record)
            if self._value(cells, self.rule_fields, "状态") != "已发布":
                continue
            rule_id = self._value(cells, self.rule_fields, "规则编号")
            city = self._value(cells, self.rule_fields, "城市")
            activity = self._value(cells, self.rule_fields, "活动名称")
            if not (rule_id and city and activity):
                continue
            source = cells.get(self.rule_fields.get("来源链接", ""))
            source_label = _cell_text(source)
            source_url = source.get("link", "") if isinstance(source, Mapping) else ""
            rules.append(
                ActivityRefundRule(
                    rule_id=rule_id,
                    city=city,
                    activity=activity,
                    card=self._value(cells, self.rule_fields, "卡项名称"),
                    period=self._value(cells, self.rule_fields, "适用购卡期间"),
                    conditions=self._value(cells, self.rule_fields, "触发条件"),
                    required_inputs=self._value(cells, self.rule_fields, "需补充信息"),
                    calculation=self._value(cells, self.rule_fields, "计算规则"),
                    exceptions=self._value(cells, self.rule_fields, "例外与限制"),
                    source_label=source_label,
                    source_url=source_url if isinstance(source_url, str) else "",
                    source_section=self._value(cells, self.rule_fields, "来源章节"),
                )
            )
        return tuple(rules)

    def _evidence(self, rules: Sequence[ActivityRefundRule]) -> tuple[Evidence, ...]:
        return tuple(
            Evidence(
                knowledge_id=rule.rule_id,
                intent=f"{rule.activity}退款规则",
                answer=rule.calculation,
                source_id=rule.source_label or "活动退款规则",
                source_section=rule.source_section,
                version=rule.period,
                risk_level="重要",
                sensitivity="限内部",
                conditions=(rule.conditions,),
                exceptions=(rule.exceptions,) if rule.exceptions else (),
                business_domain="门店运营",
                business_topic="会员服务",
                source_url=rule.source_url,
            )
            for rule in rules
        )

    def _records(self, table_id: str) -> list[object]:
        with self._lock:
            payload = self.client.query_all_records(self.base_id, table_id)
        data = payload.get("data")
        if not isinstance(data, Mapping):
            return []
        records = data.get("records")
        return list(records) if isinstance(records, list) else []

    @staticmethod
    def _cells(record: object) -> Mapping[str, object]:
        if not isinstance(record, Mapping):
            return {}
        cells = record.get("cells")
        return cells if isinstance(cells, Mapping) else {}

    @staticmethod
    def _value(cells: Mapping[str, object], fields: Mapping[str, str], name: str) -> str:
        field_id = fields.get(name, "")
        return _cell_text(cells.get(field_id)) if field_id else ""
