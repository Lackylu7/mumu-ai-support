import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mumu_kb.interpretation_migration import (
    apply_interpretation_migration_plan,
    build_interpretation_migration_plan,
)


class MigrationClient:
    def __init__(self) -> None:
        self.tables: dict[str, dict[str, object]] = {
            "tbl-knowledge": {"tableName": "知识条目", "fields": []},
        }

    def get_base(self, base_id: str):
        del base_id
        return {
            "data": {
                "tables": [
                    {"tableId": table_id, "tableName": value["tableName"]}
                    for table_id, value in self.tables.items()
                ]
            }
        }

    def get_table(self, base_id: str, table_id: str):
        del base_id
        return {
            "data": {
                "tables": [
                    {"tableId": table_id, "fields": self.tables[table_id]["fields"]}
                ]
            }
        }

    def create_table(self, base_id: str, name: str, fields):
        del base_id
        table_id = "tbl-interpretation"
        self.tables[table_id] = {
            "tableName": name,
            "fields": [
                {"fieldId": f"fld-{index}", "fieldName": item["fieldName"]}
                for index, item in enumerate(fields, start=1)
            ],
        }
        return {"data": {"tableId": table_id}}

    def create_fields(self, base_id: str, table_id: str, fields):
        del base_id
        current = self.tables[table_id]["fields"]
        start = len(current)
        current.extend(
            {"fieldId": f"fld-{start + index}", "fieldName": item["fieldName"]}
            for index, item in enumerate(fields, start=1)
        )
        return {"data": {}}


def manifest() -> dict[str, object]:
    return {
        "schema_version": 3,
        "base_name": "木木知识库",
        "base_id": "base-real",
        "tables": {"知识条目": {"table_id": "tbl-knowledge", "fields": {}}},
    }


class InterpretationMigrationTests(unittest.TestCase):
    def test_create_table_is_re_read_and_idempotent(self) -> None:
        client = MigrationClient()
        current = manifest()
        actions = build_interpretation_migration_plan(client, current)
        self.assertEqual([item.kind for item in actions], ["create_table"])

        with TemporaryDirectory() as directory:
            path = Path(directory) / ".runtime" / "aitable-manifest.json"
            updated = apply_interpretation_migration_plan(client, current, actions, path)

        self.assertEqual(updated["schema_version"], 4)
        self.assertEqual(len(updated["tables"]["制度解读"]["fields"]), 15)
        self.assertEqual(build_interpretation_migration_plan(client, updated), [])


if __name__ == "__main__":
    unittest.main()
