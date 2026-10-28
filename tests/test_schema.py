import unittest

from mumu_kb.schema import BUSINESS_DOMAINS, TABLE_SPECS, get_table_spec


class SchemaTests(unittest.TestCase):
    def test_schema_contains_interpretation_table(self) -> None:
        self.assertEqual(
            [table.name for table in TABLE_SPECS],
            [
                "知识源台账",
                "知识条目",
                "制度解读",
                "知识缺口",
                "木木测试题",
                "发布记录",
            ],
        )

    def test_interpretation_table_separates_value_and_boundaries(self) -> None:
        names = {field.name for field in get_table_spec("制度解读").fields}
        self.assertTrue(
            {
                "关联知识编号",
                "员工关切",
                "背后逻辑",
                "对员工的意义",
                "员工成本与限制",
                "推荐表达",
                "表达边界",
                "依据等级",
                "发布状态",
            }
            <= names
        )

    def test_knowledge_table_contains_publish_gate_fields(self) -> None:
        knowledge = get_table_spec("知识条目")
        names = {field.name for field in knowledge.fields}
        self.assertTrue(
            {"知识编号", "发布状态", "生效时间", "失效时间", "入库确认人", "来源编号"}
            <= names
        )

    def test_knowledge_table_has_two_level_routing_and_separate_answer_boundary(self) -> None:
        names = {field.name for field in get_table_spec("知识条目").fields}

        self.assertTrue(
            {"一级业务域", "二级主题", "信息可见范围", "回答边界", "适用制度版本"}
            <= names
        )

    def test_status_options_do_not_contain_automatic_publish(self) -> None:
        for table in TABLE_SPECS:
            for field in table.fields:
                self.assertNotIn("自动发布", field.options)

    def test_each_primary_field_is_first_and_text(self) -> None:
        for table in TABLE_SPECS:
            self.assertEqual(table.fields[0].name, table.primary_field)
            self.assertEqual(table.fields[0].field_type, "text")

    def test_case_workflow_fields_and_company_wide_domain_are_present(self) -> None:
        gap_fields = {field.name for field in get_table_spec("知识缺口").fields}

        self.assertTrue(
            {
                "原消息ID",
                "提问人ID",
                "提问人姓名",
                "案件群ID",
                "案件群名",
                "建群时间",
                "结案触发人ID",
                "结案触发人姓名",
                "结案时间",
                "自动来源编号",
            }
            <= gap_fields
        )
        self.assertEqual(
            BUSINESS_DOMAINS,
            ("教练运营", "员工合伙制", "门店运营", "人力与行政", "客户服务与品牌", "公司通用"),
        )


if __name__ == "__main__":
    unittest.main()
