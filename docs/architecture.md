# 架构与数据流

## 1. 分层结构

### 接入层

`mumu_bot.stream` 负责接收钉钉 Stream 事件，完成消息解析、@信息识别、回复发送、异常兜底和人工转接通知。外部事件先被转换成内部领域对象，避免业务逻辑直接依赖钉钉 payload。

### 编排层

`mumu_bot.service` 是核心应用服务。它负责判断普通聊天、业务问题、能力询问和人工转接意图；对多问题消息进行有限拆分并控制并发；调用知识检索、解释层和模型适配器；校验答案是否有足够证据；决定直接回答、请求补充信息或创建人工案件。

### 知识层

`mumu_kb` 提供领域模型、AI 表格访问、知识导入、审核状态、候选门禁和 payload 构造。知识内容和机器人逻辑分离，因此可以替换知识源，而不需要改动对话编排。

### 运行层

运行监督器负责进程启动、心跳、状态文件、超时、重启和 Windows 任务计划程序集成。它关注“服务是否健康”，不把业务状态混在进程生命周期里。

## 2. 一次请求的生命周期

```text
Event
  → normalize
  → session lock / deduplication
  → intent + domain route
  → retrieval plan
  → live table snapshot or cached knowledge
  → evidence normalization
  → answer policy gate
  → constrained model generation
  → citation / confidence / handoff decision
  → reply and state update
```

每一步都可以独立测试。网络访问、模型调用和钉钉发送通过接口隔离，离线测试可以使用 fake gateway 和 fake model。

## 3. 关键状态

- 会话记忆：保存必要上下文，避免把整段聊天无限拼接给模型。
- 知识版本：回答携带来源和版本信息，便于复核。
- 案件状态：`open → claimed → resolved/closed`，并记录关键消息和处理结果。
- 运行状态：记录 `starting/connecting/connected/degraded/stopped` 等状态和心跳时间。

## 4. 设计取舍

- 先检索再生成，牺牲少量延迟换取可解释性和可审计性。
- 业务问题与普通聊天分流，避免普通对话被严格知识规则限制，也避免业务问题走自由生成。
- 失败时转人工而不是继续重试模型，减少错误答案被当成制度执行的风险。
- 用小接口替代大而全的 SDK 依赖，让核心逻辑可以在无网络环境下验证。
