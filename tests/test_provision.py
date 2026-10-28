import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mumu_kb.provision import (
    ProvisionConflict,
    apply_provision_plan,
    build_provision_plan,
)


class PlanningClient:
    def __init__(self, results: list[dict[str, object]]) -> None:
        self.results = results
        self.created_bases: list[str] = []

    def search_bases(self, name: str) -> dict[str, object]:
        return {"data": {"items": self.results}}


class ApplyingClient(PlanningClient):
    def __init__(self) -> None:
        super().__init__([])
        self.base_id = "base-real"
        self.tables: dict[str, dict[str, object]] = {
            "默认表": {
                "tableId": "tbl-source",
                "name": "默认表",
                "fields": [{"fieldId": "fld-source-id", "fieldName": "标题"}],
            }
        }
        self.counter = 1

    def create_base(self, name: str) -> dict[str, object]:
        self.created_bases.append(name)
        return {"data": {"baseId": self.base_id, "baseName": name}}

    def get_base(self, base_id: str) -> dict[str, object]:
        return {
            "data": {
                "baseId": base_id,
                "tables": [
                    {"tableId": table["tableId"], "name": table["name"]}
                    for table in self.tables.values()
                ],
            }
        }

    def update_table(self, base_id: str, table_id: str, name: str) -> dict[str, object]:
        table = next(item for item in self.tables.values() if item["tableId"] == table_id)
        self.tables.pop(str(table["name"]))
        table["name"] = name
        self.tables[name] = table
        return {"data": {"tableId": table_id, "name": name}}

    def get_table(self, base_id: str, table_id: str | None = None) -> dict[str, object]:
        selected = [
            table
            for table in self.tables.values()
            if table_id is None or table["tableId"] == table_id
        ]
        return {"data": {"tables": selected}}

    def update_field(
        self, base_id: str, table_id: str, field_id: str, name: str
    ) -> dict[str, object]:
        table = next(item for item in self.tables.values() if item["tableId"] == table_id)
        field = next(item for item in table["fields"] if item["fieldId"] == field_id)
        field["fieldName"] = name
        return {"data": {"fieldId": field_id, "fieldName": name}}

    def create_fields(
        self, base_id: str, table_id: str, fields: list[dict[str, object]]
    ) -> dict[str, object]:
        table = next(item for item in self.tables.values() if item["tableId"] == table_id)
        created = []
        for definition in fields:
            self.counter += 1
            field = {
                "fieldId": f"fld-{self.counter}",
                "fieldName": definition["fieldName"],
            }
            table["fields"].append(field)
            created.append(field)
        return {"data": {"fields": created}}

    def create_table(
        self, base_id: str, name: str, fields: list[dict[str, object]]
    ) -> dict[str, object]:
        table_id = f"tbl-{len(self.tables) + 1}"
        table_fields = []
        for definition in fields:
            self.counter += 1
            table_fields.append(
                {
                    "fieldId": f"fld-{self.counter}",
                    "fieldName": definition["fieldName"],
                }
            )
        self.tables[name] = {
            "tableId": table_id,
            "name": name,
            "fields": table_fields,
        }
        return {"data": {"tableId": table_id, "name": name}}


class ProvisionTests(unittest.TestCase):
    def test_existing_same_name_base_stops_without_creating(self) -> None:
        client = PlanningClient([{"baseId": "base-real", "baseName": "木木知识库"}])

        with self.assertRaisesRegex(ProvisionConflict, "base-real"):
            build_provision_plan(client)

        self.assertEqual(client.created_bases, [])

    def test_new_base_reuses_default_table_as_source_registry(self) -> None:
        plan = build_provision_plan(PlanningClient([]))

        self.assertEqual(plan[0].kind, "create_base")
        self.assertTrue(
            any(
                action.kind == "rename_default_table"
                and action.table == "知识源台账"
                for action in plan
            )
        )
        self.assertFalse(any(action.kind == "delete_table" for action in plan))

    def test_apply_writes_complete_manifest_under_runtime(self) -> None:
        client = ApplyingClient()
        plan = build_provision_plan(client)
        with TemporaryDirectory() as directory:
            runtime = Path(directory) / ".runtime"
            manifest_path = runtime / "aitable-manifest.json"

            manifest = apply_provision_plan(client, plan, manifest_path)

            self.assertEqual(manifest["base_id"], "base-real")
            self.assertEqual(manifest["schema_version"], 5)
            self.assertEqual(set(manifest["tables"]), {
                "知识源台账", "知识条目", "制度解读", "知识缺口", "木木测试题", "发布记录"
            })
            self.assertEqual(
                manifest["tables"]["知识源台账"]["fields"]["来源编号"],
                "fld-source-id",
            )
            self.assertTrue(manifest_path.exists())


if __name__ == "__main__":
    unittest.main()
