from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Protocol


class DwsError(RuntimeError):
    pass


class CommandRunner(Protocol):
    def run(self, args: list[str]) -> subprocess.CompletedProcess[str]: ...


class SubprocessRunner:
    def run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            shell=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            # dws.cmd otherwise creates a visible console window on Windows.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )


class DwsClient:
    def __init__(
        self,
        profile: str,
        runner: CommandRunner | None = None,
        executable: str | None = None,
    ) -> None:
        if not profile.strip():
            raise ValueError("DWS profile 不能为空")
        self.profile = profile
        self.runner = runner or SubprocessRunner()
        if executable:
            self.executable = executable
        elif runner is not None:
            self.executable = "dws"
        elif os.name == "nt":
            self.executable = shutil.which("dws.cmd") or shutil.which("dws") or "dws.cmd"
        else:
            self.executable = shutil.which("dws") or "dws"

    def run_json(
        self,
        args: list[str],
        *,
        timeout_seconds: int | None = None,
    ) -> dict[str, object]:
        if "--format" in args:
            raise ValueError("run_json 统一管理 --format 参数")
        command = [self.executable, "--profile", self.profile, *args, "--format", "json"]
        if timeout_seconds is not None:
            command.extend(("--timeout", str(timeout_seconds)))
        result = self.runner.run(command)
        if result.returncode != 0:
            stderr = " ".join(result.stderr.strip().splitlines())[:800]
            raise DwsError(f"DWS 退出码 {result.returncode}: {stderr}")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise DwsError("DWS 没有返回合法 JSON") from error
        if not isinstance(payload, dict):
            raise DwsError("DWS JSON 顶层必须是对象")
        return payload

    def search_bases(self, name: str) -> dict[str, object]:
        return self.run_json(["aitable", "base", "search", "--query", name])

    def search_wiki_nodes(
        self,
        workspace_id: str,
        query: str,
        *,
        limit: int = 8,
    ) -> dict[str, object]:
        if not workspace_id.strip() or not query.strip():
            raise ValueError("知识库 ID 和搜索问题不能为空")
        if not 1 <= limit <= 30:
            raise ValueError("知识库搜索条数必须在 1 到 30 之间")
        return self.run_json(
            [
                "wiki",
                "node",
                "search",
                "--workspace",
                workspace_id,
                "--query",
                query,
                "--extensions",
                "adoc",
                "--limit",
                str(limit),
            ]
        )

    def read_doc(self, node_id: str) -> dict[str, object]:
        if not node_id.strip():
            raise ValueError("文档节点 ID 不能为空")
        return self.run_json(["doc", "read", "--node", node_id])

    def create_base(self, name: str) -> dict[str, object]:
        return self.run_json(["aitable", "base", "create", "--name", name])

    def get_base(self, base_id: str) -> dict[str, object]:
        return self.run_json(["aitable", "base", "get", "--base-id", base_id])

    def get_table(self, base_id: str, table_id: str | None = None) -> dict[str, object]:
        args = ["aitable", "table", "get", "--base-id", base_id]
        if table_id:
            args.extend(["--table-ids", table_id])
        return self.run_json(args)

    def create_table(
        self,
        base_id: str,
        name: str,
        fields: list[dict[str, object]],
    ) -> dict[str, object]:
        return self.run_json(
            [
                "aitable",
                "table",
                "create",
                "--base-id",
                base_id,
                "--name",
                name,
                "--fields",
                json.dumps(fields, ensure_ascii=False, separators=(",", ":")),
            ]
        )

    def update_table(self, base_id: str, table_id: str, name: str) -> dict[str, object]:
        return self.run_json(
            [
                "aitable",
                "table",
                "update",
                "--base-id",
                base_id,
                "--table-id",
                table_id,
                "--name",
                name,
            ]
        )

    def create_fields(
        self,
        base_id: str,
        table_id: str,
        fields: list[dict[str, object]],
    ) -> dict[str, object]:
        return self.run_json(
            [
                "aitable",
                "field",
                "create",
                "--base-id",
                base_id,
                "--table-id",
                table_id,
                "--fields",
                json.dumps(fields, ensure_ascii=False, separators=(",", ":")),
            ]
        )

    def get_fields(
        self,
        base_id: str,
        table_id: str,
        field_ids: tuple[str, ...] = (),
    ) -> dict[str, object]:
        args = [
            "aitable",
            "field",
            "get",
            "--base-id",
            base_id,
            "--table-id",
            table_id,
        ]
        if field_ids:
            args.extend(["--field-ids", ",".join(field_ids)])
        return self.run_json(args)

    def update_field(
        self,
        base_id: str,
        table_id: str,
        field_id: str,
        *,
        name: str | None = None,
        config: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if name is None and config is None:
            raise ValueError("字段名称和配置至少提供一项")
        args = [
            "aitable",
            "field",
            "update",
            "--base-id",
            base_id,
            "--table-id",
            table_id,
            "--field-id",
            field_id,
        ]
        if name is not None:
            args.extend(["--name", name])
        if config is not None:
            args.extend(
                [
                    "--config",
                    json.dumps(config, ensure_ascii=False, separators=(",", ":")),
                ]
            )
        return self.run_json(args)

    def create_records(
        self,
        base_id: str,
        table_id: str,
        records_file: Path,
    ) -> dict[str, object]:
        return self.run_json(
            [
                "aitable",
                "record",
                "create",
                "--base-id",
                base_id,
                "--table-id",
                table_id,
                "--records-file",
                str(records_file),
            ]
        )

    def update_records(
        self,
        base_id: str,
        table_id: str,
        records_file: Path,
    ) -> dict[str, object]:
        return self.run_json(
            [
                "aitable",
                "record",
                "update",
                "--base-id",
                base_id,
                "--table-id",
                table_id,
                "--records-file",
                str(records_file),
            ]
        )

    def create_chat_group(self, name: str, users: str) -> dict[str, object]:
        return self.run_json(
            [
                "chat",
                "group",
                "create",
                "--name",
                name,
                "--users",
                users,
                "--type",
                "INTERNAL",
            ]
        )

    def search_chat_groups(
        self,
        query: str,
        *,
        limit: int = 20,
        cursor: str = "0",
    ) -> dict[str, object]:
        return self.run_json(
            [
                "chat",
                "search",
                "--query",
                query,
                "--limit",
                str(limit),
                "--cursor",
                cursor,
            ]
        )

    def add_bot_to_group(
        self,
        group_id: str,
        robot_code: str,
    ) -> dict[str, object]:
        return self.run_json(
            [
                "chat",
                "group",
                "members",
                "add-bot",
                "--robot-code",
                robot_code,
                "--id",
                group_id,
            ]
        )

    def list_group_bots(self, group_id: str) -> dict[str, object]:
        return self.run_json(
            [
                "chat",
                "group",
                "bots",
                "--group",
                group_id,
            ]
        )

    def send_chat_message_by_bot(
        self,
        *,
        robot_code: str,
        group_id: str,
        title: str,
        text: str,
        at_user_ids: tuple[str, ...] = (),
    ) -> dict[str, object]:
        args = [
            "chat",
            "message",
            "send-by-bot",
            "--robot-code",
            robot_code,
            "--group",
            group_id,
            "--title",
            title,
            "--text",
            text,
        ]
        if at_user_ids:
            args.extend(["--at-user-ids", ",".join(at_user_ids)])
        return self.run_json(args)

    def list_chat_messages(
        self,
        *,
        group_id: str,
        start_time: str,
        direction: str,
        limit: int,
    ) -> dict[str, object]:
        return self.run_json(
            [
                "chat",
                "message",
                "list",
                "--group",
                group_id,
                "--time",
                start_time,
                "--direction",
                direction,
                "--limit",
                str(limit),
            ]
        )

    def add_message_emoji(
        self,
        conversation_id: str,
        message_id: str,
        emoji: str,
        *,
        timeout_seconds: int | None = None,
    ) -> dict[str, object]:
        return self.run_json(
            [
                "chat",
                "message",
                "add-emoji",
                "--conversation-id",
                conversation_id,
                "--msg-id",
                message_id,
                "--emoji",
                emoji,
            ],
            timeout_seconds=timeout_seconds,
        )

    def remove_message_emoji(
        self,
        conversation_id: str,
        message_id: str,
        emoji: str,
        *,
        timeout_seconds: int | None = None,
    ) -> dict[str, object]:
        return self.run_json(
            [
                "chat",
                "message",
                "remove-emoji",
                "--conversation-id",
                conversation_id,
                "--msg-id",
                message_id,
                "--emoji",
                emoji,
            ],
            timeout_seconds=timeout_seconds,
        )

    def create_todo_task(
        self,
        *,
        title: str,
        executor_user_ids: tuple[str, ...],
        priority: str = "30",
    ) -> dict[str, object]:
        if not executor_user_ids:
            raise ValueError("待办执行人不能为空")
        return self.run_json(
            [
                "todo",
                "task",
                "create",
                "--title",
                title,
                "--executors",
                ",".join(executor_user_ids),
                "--priority",
                priority,
            ]
        )

    def add_todo_comment(self, task_id: str, content: str) -> dict[str, object]:
        return self.run_json(
            [
                "todo",
                "comment",
                "add",
                "--task-id",
                task_id,
                "--content",
                content,
            ]
        )

    def complete_todo_task(self, task_id: str) -> dict[str, object]:
        return self.run_json(
            [
                "todo",
                "task",
                "update",
                "--task-id",
                task_id,
                "--done",
                "true",
            ]
        )

    def query_all_records(self, base_id: str, table_id: str) -> dict[str, object]:
        # Do pagination here instead of using the CLI's --all composite.  The
        # latter treats a valid terminal empty page as an incomplete response
        # for some tables, which makes background scanners fail repeatedly.
        records: list[object] = []
        cursor = ""
        seen_cursors: set[str] = set()
        while True:
            args = [
                "aitable",
                "record",
                "query",
                "--base-id",
                base_id,
                "--table-id",
                table_id,
                "--limit",
                "100",
            ]
            if cursor:
                args.extend(["--cursor", cursor])
            payload = self.run_json(args)
            data = payload.get("data")
            if not isinstance(data, dict):
                raise DwsError("DWS 查询记录返回缺少 data")
            page = data.get("records")
            if page is not None:
                if not isinstance(page, list):
                    raise DwsError("DWS 查询记录 records 必须是数组")
                records.extend(page)
            next_cursor = data.get("nextCursor")
            if not isinstance(next_cursor, str) or not next_cursor:
                break
            if next_cursor in seen_cursors:
                raise DwsError("DWS 查询记录分页游标重复")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        return {"data": {"records": records}}

    def query_records(
        self,
        *,
        base_id: str,
        table_id: str,
        field_ids: list[str],
        filters: dict[str, object],
        query: str,
        limit: int,
    ) -> dict[str, object]:
        args = [
            "aitable",
            "record",
            "query",
            "--base-id",
            base_id,
            "--table-id",
            table_id,
            "--field-ids",
            ",".join(field_ids),
            "--filters",
            json.dumps(filters, ensure_ascii=False, separators=(",", ":")),
        ]
        if query:
            args.extend(["--query", query])
        args.extend(["--limit", str(limit)])
        return self.run_json(args)
