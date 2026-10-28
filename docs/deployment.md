# 本地部署与配置

## 1. 环境要求

- Windows 10/11 或其他可运行 Python 3.11+ 的环境
- Python 3.11+
- 可访问的钉钉 Stream 应用（真实部署时）
- 已授权的 AI 表格/知识库（真实部署时）
- DeepSeek-compatible API（真实部署时）

## 2. 安装

```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -e ".[test]"
Copy-Item .env.example .env
```

只做离线测试时不需要填写真实账号。真实运行前逐项填写 `.env`，并确认 `.env` 没有被 Git 跟踪。

## 3. 运行命令

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe -m mumu_bot.cli --help
.\\.venv\\Scripts\\python.exe -m mumu_kb.cli --help
.\\scripts\\start-mumu-local.ps1 -Preview
.\\scripts\\status-mumu-local.ps1
```

真实启动命令和参数以 CLI 帮助为准；不要把凭据直接写入命令行，因为命令行可能被进程查看工具记录。

## 4. 配置原则

- API key、client secret 等只放在本地环境变量或受控密钥管理器中。
- `MUMU_ALLOWED_USER_IDS` 使用最小授权集合。
- `MUMU_CASE_WORKFLOW_ENABLED=true` 时，必须同时提供案件负责人配置。
- `MUMU_RETRIEVAL_MODE=v2_live` 适合需要实时知识的场景；离线演示应使用合成 fixture。
- 生产环境应为 `.runtime` 和日志目录设置访问控制与保留期限。

## 5. 平台边界

仓库提供 Windows 本地运行脚本。Mac 原生打包、钉钉线上验收和企业知识库接入需要在目标环境中单独验证；本仓库不把未验证的部署结果写成已完成事实。
