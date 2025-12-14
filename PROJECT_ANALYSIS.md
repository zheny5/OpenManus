# OpenManus 项目深度技术拆解

> 本文档提供 OpenManus 项目的全面技术分析，包括架构设计、核心模块、实现细节和使用指南。

## 目录

- [一、项目概述](#一项目概述)
- [二、整体架构](#二整体架构)
- [三、核心模块详解](#三核心模块详解)
- [四、数据流与交互](#四数据流与交互)
- [五、关键技术实现](#五关键技术实现)
- [六、扩展开发指南](#六扩展开发指南)
- [七、配置与部署](#七配置与部署)
- [八、最佳实践](#八最佳实践)

---

## 一、项目概述

### 1.1 项目定位

OpenManus 是一个**开源通用 AI 智能体框架**，旨在提供一个无需邀请码、功能完整的智能体开发平台。项目由 MetaGPT 团队开发，支持多种工具集成和任务执行模式。

### 1.2 核心特性

- ✅ **通用智能体**：支持多种任务类型的智能体实现
- ✅ **工具生态**：丰富的内置工具和可扩展的 MCP 工具支持
- ✅ **多智能体协作**：支持规划流程和多智能体协同工作
- ✅ **安全隔离**：支持沙箱环境执行代码
- ✅ **浏览器自动化**：完整的网页交互能力
- ✅ **灵活配置**：支持多种 LLM 提供商和配置方式

### 1.3 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.12+ |
| 框架 | Pydantic, asyncio |
| LLM | OpenAI, Azure OpenAI, Anthropic, Google, Ollama |
| 浏览器 | Playwright, browser-use |
| 代码执行 | Docker, multiprocessing |
| 配置管理 | TOML, JSON |
| 搜索 | Google, DuckDuckGo, Baidu, Bing |

---

## 二、整体架构

### 2.1 架构层次图

```
┌─────────────────────────────────────────────────────────┐
│                    用户接口层                              │
│  main.py / run_flow.py / run_mcp.py / sandbox_main.py   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                    智能体层 (Agent Layer)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │  Manus   │  │DataAnalysis│ │ Browser │ │  MCP    │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬────┘ │
│       │             │              │             │       │
│  ┌────▼─────────────▼──────────────▼─────────────▼────┐ │
│  │         ToolCallAgent (工具调用基类)                  │ │
│  └────────────────────┬────────────────────────────────┘ │
│                       │                                    │
│  ┌────────────────────▼────────────────────────────────┐ │
│  │            ReActAgent (ReAct 模式)                   │ │
│  └────────────────────┬────────────────────────────────┘ │
│                       │                                    │
│  ┌────────────────────▼────────────────────────────────┐ │
│  │            BaseAgent (抽象基类)                      │ │
│  └─────────────────────────────────────────────────────┘ │
└────────────────────┬──────────────────────────────────────┘
                     │
┌────────────────────▼──────────────────────────────────────┐
│                    工具层 (Tool Layer)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │PythonExecute │  │BrowserUseTool │  │  WebSearch   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │StrReplaceEdit│  │ FileOperators│  │   MCP Tools  │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└────────────────────┬──────────────────────────────────────┘
                     │
┌────────────────────▼──────────────────────────────────────┐
│                    基础设施层 (Infrastructure)              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │   LLM    │  │  Config  │  │  Memory  │  │ Sandbox │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
└───────────────────────────────────────────────────────────┘
```

### 2.2 目录结构详解

```
OpenManus/
├── app/                          # 核心应用代码
│   ├── agent/                    # 智能体实现
│   │   ├── base.py              # 基础智能体抽象类
│   │   ├── react.py             # ReAct 模式实现
│   │   ├── toolcall.py          # 工具调用智能体
│   │   ├── manus.py             # Manus 通用智能体
│   │   ├── data_analysis.py     # 数据分析智能体
│   │   ├── browser.py           # 浏览器智能体
│   │   ├── mcp.py               # MCP 协议智能体
│   │   ├── swe.py               # 软件工程智能体
│   │   └── sandbox_agent.py     # 沙箱智能体
│   │
│   ├── tool/                     # 工具集合
│   │   ├── base.py              # 工具基类
│   │   ├── python_execute.py   # Python 代码执行
│   │   ├── browser_use_tool.py  # 浏览器自动化
│   │   ├── web_search.py        # 网络搜索
│   │   ├── str_replace_editor.py # 文本编辑器
│   │   ├── file_operators.py    # 文件操作
│   │   ├── mcp.py               # MCP 工具集成
│   │   ├── planning.py          # 规划工具
│   │   ├── search/              # 搜索引擎实现
│   │   ├── sandbox/             # 沙箱工具
│   │   └── chart_visualization/ # 数据可视化
│   │
│   ├── flow/                     # 流程管理
│   │   ├── base.py              # 流程基类
│   │   ├── planning.py          # 规划流程
│   │   └── flow_factory.py      # 流程工厂
│   │
│   ├── mcp/                      # MCP 协议支持
│   │   └── server.py            # MCP 服务器实现
│   │
│   ├── sandbox/                  # 沙箱环境
│   │   └── client.py            # 沙箱客户端
│   │
│   ├── prompt/                   # 提示词模板
│   │   ├── manus.py             # Manus 提示词
│   │   ├── toolcall.py          # 工具调用提示词
│   │   ├── planning.py          # 规划提示词
│   │   └── ...
│   │
│   ├── schema.py                 # 数据模型定义
│   ├── config.py                 # 配置管理
│   ├── llm.py                    # LLM 接口封装
│   ├── logger.py                 # 日志系统
│   └── exceptions.py             # 异常定义
│
├── config/                       # 配置文件
│   ├── config.example.toml      # 配置示例
│   └── mcp.example.json          # MCP 配置示例
│
├── workspace/                    # 工作空间（用户文件）
├── examples/                     # 示例代码
├── tests/                        # 测试代码
├── main.py                       # 主入口
├── run_flow.py                   # 多智能体流程
├── run_mcp.py                    # MCP 模式
└── requirements.txt             # 依赖列表
```

---

## 三、核心模块详解

### 3.1 智能体系统 (Agent System)

#### 3.1.1 BaseAgent - 基础智能体类

**位置**: `app/agent/base.py`

**核心功能**:
- 状态管理：IDLE → RUNNING → FINISHED/ERROR
- 记忆管理：保存对话历史和工具调用记录
- 执行循环：控制智能体的执行步骤
- 卡死检测：防止智能体陷入循环

**关键方法**:

```python
class BaseAgent(BaseModel, ABC):
    # 核心属性
    name: str                          # 智能体名称
    description: Optional[str]         # 智能体描述
    system_prompt: Optional[str]       # 系统提示词
    next_step_prompt: Optional[str]   # 下一步提示词
    llm: LLM                          # LLM 实例
    memory: Memory                    # 记忆存储
    state: AgentState                 # 当前状态
    max_steps: int = 10               # 最大步数
    current_step: int = 0             # 当前步数
    
    async def run(self, request: Optional[str] = None) -> str:
        """执行智能体的主循环"""
        # 1. 验证状态
        # 2. 添加用户请求到记忆
        # 3. 进入运行状态
        # 4. 循环执行 step() 直到完成或达到最大步数
        # 5. 返回执行结果
    
    @abstractmethod
    async def step(self) -> str:
        """执行单步操作（子类必须实现）"""
        pass
    
    def is_stuck(self) -> bool:
        """检测是否卡死（重复响应）"""
        # 检查最近的消息是否有重复内容
        # 如果重复次数 >= duplicate_threshold，返回 True
```

**状态转换**:

```
IDLE ──(run)──> RUNNING ──(完成)──> FINISHED
  │                              │
  │                              └──(错误)──> ERROR
  └──(中断)───────────────────────────────> IDLE
```

#### 3.1.2 ReActAgent - ReAct 模式实现

**位置**: `app/agent/react.py`

**设计模式**: ReAct (Reasoning + Acting)

**执行流程**:

```
1. think() - 思考下一步行动
   └─> 调用 LLM 分析当前状态
   └─> 决定是否需要执行动作
   
2. act() - 执行动作
   └─> 根据思考结果执行相应操作
   └─> 返回执行结果
   
3. step() - 组合 think 和 act
   └─> step() = think() + act()
```

**关键实现**:

```python
class ReActAgent(BaseAgent, ABC):
    @abstractmethod
    async def think(self) -> bool:
        """思考并决定下一步行动"""
        pass
    
    @abstractmethod
    async def act(self) -> str:
        """执行决定的动作"""
        pass
    
    async def step(self) -> str:
        """执行单步：思考 + 行动"""
        should_act = await self.think()
        if not should_act:
            return "Thinking complete - no action needed"
        return await self.act()
```

#### 3.1.3 ToolCallAgent - 工具调用智能体

**位置**: `app/agent/toolcall.py`

**核心功能**:
- 工具选择：LLM 根据任务选择合适的工具
- 工具执行：异步执行选定的工具
- 结果处理：将工具结果格式化并加入记忆
- 错误处理：处理工具执行失败的情况

**执行流程**:

```python
async def think(self) -> bool:
    """思考并选择工具"""
    # 1. 构建消息列表（包含历史对话）
    # 2. 调用 LLM 的 ask_tool() 方法
    # 3. LLM 返回工具调用列表
    # 4. 解析工具调用并保存到 self.tool_calls
    # 5. 返回是否需要执行动作

async def act(self) -> str:
    """执行选定的工具"""
    # 1. 遍历 self.tool_calls
    # 2. 从 available_tools 中找到对应工具
    # 3. 解析工具参数（JSON）
    # 4. 异步执行工具
    # 5. 将结果格式化为 ToolResult
    # 6. 添加到记忆（tool message）
    # 7. 检查是否有终止工具被调用
```

**工具调用格式**:

```json
{
  "id": "call_xxx",
  "type": "function",
  "function": {
    "name": "python_execute",
    "arguments": "{\"code\": \"print('Hello')\"}"
  }
}
```

#### 3.1.4 Manus - 通用智能体

**位置**: `app/agent/manus.py`

**特性**:
- 集成多种内置工具
- 支持 MCP 工具扩展
- 浏览器上下文管理
- 动态工具加载

**内置工具**:

```python
available_tools = ToolCollection(
    PythonExecute(),      # Python 代码执行
    BrowserUseTool(),     # 浏览器自动化
    StrReplaceEditor(),   # 文本编辑
    AskHuman(),           # 人机交互
    Terminate(),          # 终止执行
)
```

**MCP 集成**:

```python
async def initialize_mcp_servers(self):
    """初始化 MCP 服务器连接"""
    for server_id, server_config in config.mcp_config.servers.items():
        if server_config.type == "sse":
            # SSE 连接
            await self.connect_mcp_server(server_config.url, server_id)
        elif server_config.type == "stdio":
            # 标准输入输出连接
            await self.connect_mcp_server(
                server_config.command,
                server_id,
                use_stdio=True,
                stdio_args=server_config.args,
            )
```

**浏览器上下文管理**:

```python
async def think(self) -> bool:
    """处理浏览器上下文"""
    # 检查最近是否使用了浏览器工具
    browser_in_use = any(
        tc.function.name == BrowserUseTool().name
        for msg in recent_messages
        if msg.tool_calls
        for tc in msg.tool_calls
    )
    
    if browser_in_use:
        # 使用浏览器上下文提示词
        self.next_step_prompt = (
            await self.browser_context_helper.format_next_step_prompt()
        )
    
    return await super().think()
```

### 3.2 工具系统 (Tool System)

#### 3.2.1 BaseTool - 工具基类

**位置**: `app/tool/base.py`

**核心结构**:

```python
class BaseTool(ABC, BaseModel):
    name: str                    # 工具名称（唯一标识）
    description: str             # 工具描述（用于 LLM 选择）
    parameters: Optional[dict]  # 参数模式（JSON Schema）
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """执行工具（子类必须实现）"""
        pass
    
    def to_param(self) -> Dict:
        """转换为 OpenAI Function Calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
    
    def success_response(self, data: Union[Dict[str, Any], str]) -> ToolResult:
        """创建成功响应"""
        return ToolResult(output=text)
    
    def fail_response(self, msg: str) -> ToolResult:
        """创建失败响应"""
        return ToolResult(error=msg)
```

**ToolResult 结构**:

```python
class ToolResult(BaseModel):
    output: Optional[str] = None          # 工具输出
    error: Optional[str] = None           # 错误信息
    base64_image: Optional[str] = None    # 图片（base64）
    system: Optional[str] = None          # 系统消息
```

#### 3.2.2 PythonExecute - Python 代码执行工具

**位置**: `app/tool/python_execute.py`

**安全机制**:
- 使用 `multiprocessing` 隔离执行
- 超时控制（默认 5 秒）
- 限制内置函数访问
- 捕获 stdout 输出

**实现细节**:

```python
async def execute(self, code: str, timeout: int = 5) -> Dict:
    """执行 Python 代码"""
    with multiprocessing.Manager() as manager:
        result = manager.dict({"observation": "", "success": False})
        
        # 创建安全的全局环境
        safe_globals = {"__builtins__": __builtins__.__dict__.copy()}
        
        # 在独立进程中执行
        proc = multiprocessing.Process(
            target=self._run_code, args=(code, result, safe_globals)
        )
        proc.start()
        proc.join(timeout)
        
        # 超时处理
        if proc.is_alive():
            proc.terminate()
            proc.join(1)
            return {"observation": f"Execution timeout after {timeout} seconds", 
                    "success": False}
        
        return dict(result)
```

**注意事项**:
- 只捕获 `print()` 输出，不捕获返回值
- 使用 `StringIO` 重定向 stdout
- 异常会被捕获并返回错误信息

#### 3.2.3 BrowserUseTool - 浏览器自动化工具

**位置**: `app/tool/browser_use_tool.py`

**支持的操作**:

```python
actions = [
    "go_to_url",           # 访问 URL
    "click_element",       # 点击元素
    "input_text",          # 输入文本
    "scroll_down",         # 向下滚动
    "scroll_up",           # 向上滚动
    "scroll_to_text",      # 滚动到文本
    "send_keys",           # 发送键盘事件
    "get_dropdown_options", # 获取下拉选项
    "select_dropdown_option", # 选择下拉选项
    "go_back",            # 返回上一页
    "web_search",         # 网页搜索
    "wait",               # 等待
    "extract_content",    # 提取内容
    "switch_tab",         # 切换标签页
    "open_tab",           # 打开标签页
    "close_tab",          # 关闭标签页
]
```

**浏览器上下文管理**:

```python
class BrowserContextHelper:
    """管理浏览器上下文和状态"""
    
    async def format_next_step_prompt(self) -> str:
        """格式化下一步提示词，包含当前页面信息"""
        # 1. 获取当前页面截图
        # 2. 提取页面 DOM 结构
        # 3. 分析可交互元素
        # 4. 生成上下文提示词
```

**配置选项**:

```toml
[browser]
headless = false                    # 是否无头模式
disable_security = true             # 禁用安全特性
extra_chromium_args = []            # 额外 Chrome 参数
chrome_instance_path = null         # Chrome 实例路径
wss_url = null                      # WebSocket 连接 URL
cdp_url = null                      # CDP 连接 URL
proxy = {server = "...", ...}       # 代理配置
max_content_length = 2000          # 最大内容长度
```

#### 3.2.4 WebSearch - 网络搜索工具

**位置**: `app/tool/web_search.py`

**支持的搜索引擎**:
- Google Search
- DuckDuckGo
- Baidu
- Bing

**搜索配置**:

```toml
[search]
engine = "Google"                    # 主搜索引擎
fallback_engines = ["DuckDuckGo", "Baidu", "Bing"]  # 备用引擎
retry_delay = 60                     # 重试延迟（秒）
max_retries = 3                      # 最大重试次数
lang = "en"                          # 语言代码
country = "us"                       # 国家代码
```

**搜索流程**:

```python
async def execute(self, query: str) -> ToolResult:
    """执行搜索"""
    # 1. 尝试主搜索引擎
    # 2. 如果失败，尝试备用引擎
    # 3. 如果所有引擎都失败，等待后重试
    # 4. 返回搜索结果
```

#### 3.2.5 MCP 工具集成

**位置**: `app/tool/mcp.py`

**MCP (Model Context Protocol)** 是一个标准协议，用于连接外部工具服务器。

**连接方式**:

1. **SSE (Server-Sent Events)**:
```python
await mcp_clients.connect_sse(server_url, server_id)
```

2. **STDIO (Standard Input/Output)**:
```python
await mcp_clients.connect_stdio(command, args, server_id)
```

**MCP 配置** (`config/mcp.json`):

```json
{
  "mcpServers": {
    "filesystem": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
    },
    "github": {
      "type": "sse",
      "url": "http://localhost:8000/sse"
    }
  }
}
```

**工具动态加载**:

```python
# 连接 MCP 服务器后，自动加载其工具
new_tools = [
    tool for tool in self.mcp_clients.tools 
    if tool.server_id == server_id
]
self.available_tools.add_tools(*new_tools)
```

### 3.3 流程管理系统 (Flow System)

#### 3.3.1 BaseFlow - 流程基类

**位置**: `app/flow/base.py`

**核心功能**:
- 多智能体管理
- 智能体选择与调度
- 流程执行控制

**智能体管理**:

```python
class BaseFlow(BaseModel, ABC):
    agents: Dict[str, BaseAgent]      # 智能体字典
    primary_agent_key: Optional[str]  # 主智能体键
    
    def get_agent(self, key: str) -> Optional[BaseAgent]:
        """获取指定智能体"""
        return self.agents.get(key)
    
    def add_agent(self, key: str, agent: BaseAgent) -> None:
        """添加新智能体"""
        self.agents[key] = agent
    
    @abstractmethod
    async def execute(self, input_text: str) -> str:
        """执行流程"""
        pass
```

#### 3.3.2 PlanningFlow - 规划流程

**位置**: `app/flow/planning.py`

**规划流程**:

```
1. 任务分解
   └─> 使用 PlanningTool 将任务分解为步骤
   └─> 每个步骤包含：描述、状态、执行者

2. 步骤执行
   └─> 遍历步骤列表
   └─> 根据步骤类型选择执行者
   └─> 执行步骤并更新状态

3. 动态调整
   └─> 根据执行结果调整计划
   └─> 处理阻塞步骤
   └─> 重新规划失败步骤
```

**步骤状态**:

```python
class PlanStepStatus(str, Enum):
    NOT_STARTED = "not_started"    # 未开始
    IN_PROGRESS = "in_progress"     # 进行中
    COMPLETED = "completed"         # 已完成
    BLOCKED = "blocked"             # 阻塞
```

**执行流程**:

```python
async def execute(self, input_text: str) -> str:
    """执行规划流程"""
    # 1. 创建或加载计划
    plan = await self.create_or_load_plan(input_text)
    
    # 2. 执行计划步骤
    while not self.is_plan_complete(plan):
        # 获取下一个待执行步骤
        step = self.get_next_step(plan)
        
        # 选择执行者
        executor = self.get_executor(step.type)
        
        # 执行步骤
        result = await executor.run(step.description)
        
        # 更新步骤状态
        self.update_step_status(step, result)
        
        # 检查是否需要重新规划
        if self.needs_replanning(plan):
            plan = await self.replan(plan)
    
    return self.format_final_result(plan)
```

### 3.4 LLM 接口系统

#### 3.4.1 LLM 类

**位置**: `app/llm.py`

**核心功能**:
- 多提供商支持（OpenAI, Azure, Anthropic, Google, Ollama）
- Token 计数与管理
- 重试机制
- 工具调用支持
- 多模态支持（文本 + 图片）

**支持的 API 类型**:

```python
api_types = {
    "OpenAI": AsyncOpenAI,
    "Azure": AsyncAzureOpenAI,
    "Anthropic": AsyncAnthropic,
    "Google": AsyncGoogleGenerativeAI,
    "Ollama": AsyncOpenAI,  # 兼容 OpenAI 格式
}
```

**Token 计数**:

```python
class TokenCounter:
    """精确计算 Token 数量"""
    
    def count_message_tokens(self, messages: List[dict]) -> int:
        """计算消息列表的 Token 数"""
        # 1. 基础格式 Token（2）
        # 2. 每条消息 Token（4）
        # 3. 文本内容 Token
        # 4. 图片 Token（根据细节级别）
        # 5. 工具调用 Token
```

**重试机制**:

```python
@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
)
async def ask(self, messages: List[Message], ...):
    """带重试的 LLM 调用"""
    pass
```

**工具调用**:

```python
async def ask_tool(
    self,
    messages: List[Message],
    tools: List[dict],
    tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,
) -> ChatCompletionMessage:
    """调用 LLM 并支持工具选择"""
    # 1. 构建工具参数列表
    # 2. 调用 Chat Completion API
    # 3. 解析工具调用结果
    # 4. 返回消息对象
```

### 3.5 配置管理系统

#### 3.5.1 Config 类

**位置**: `app/config.py`

**配置结构**:

```python
class AppConfig(BaseModel):
    llm: Dict[str, LLMSettings]           # LLM 配置（支持多个）
    sandbox: Optional[SandboxSettings]    # 沙箱配置
    browser_config: Optional[BrowserSettings]  # 浏览器配置
    search_config: Optional[SearchSettings]   # 搜索配置
    mcp_config: Optional[MCPSettings]     # MCP 配置
    run_flow_config: Optional[RunflowSettings]  # 流程配置
    daytona_config: Optional[DaytonaSettings]   # Daytona 配置
```

**配置加载**:

```python
class Config:
    """单例配置类"""
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        # 1. 查找配置文件（config.toml 或 config.example.toml）
        # 2. 解析 TOML 文件
        # 3. 加载 MCP 配置（mcp.json）
        # 4. 构建配置对象
```

**配置文件示例** (`config/config.toml`):

```toml
# 全局 LLM 配置
[llm]
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."
max_tokens = 4096
temperature = 0.0
api_type = "OpenAI"
api_version = ""

# 特定用途的 LLM 配置（可选）
[llm.vision]
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."

# 浏览器配置
[browser]
headless = false
disable_security = true
max_content_length = 2000

# 搜索配置
[search]
engine = "Google"
fallback_engines = ["DuckDuckGo", "Baidu", "Bing"]
lang = "en"
country = "us"

# 沙箱配置
[sandbox]
use_sandbox = false
image = "python:3.12-slim"
work_dir = "/workspace"
memory_limit = "512m"
cpu_limit = 1.0
timeout = 300
network_enabled = false

# MCP 配置
[mcp]
server_reference = "app.mcp.server"

# 流程配置
[runflow]
use_data_analysis_agent = false
```

### 3.6 数据模型系统

#### 3.6.1 Schema 定义

**位置**: `app/schema.py`

**核心模型**:

```python
# 消息角色
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

# 智能体状态
class AgentState(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"

# 工具选择
class ToolChoice(str, Enum):
    NONE = "none"        # 不使用工具
    AUTO = "auto"        # 自动选择
    REQUIRED = "required" # 必须使用工具

# 消息模型
class Message(BaseModel):
    role: ROLE_TYPE
    content: Optional[str]
    tool_calls: Optional[List[ToolCall]]
    name: Optional[str]
    tool_call_id: Optional[str]
    base64_image: Optional[str]

# 记忆模型
class Memory(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = 100
    
    def add_message(self, message: Message) -> None:
        """添加消息到记忆"""
        self.messages.append(message)
        # 限制消息数量
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
```

---

## 四、数据流与交互

### 4.1 单智能体执行流程

```
用户输入
   │
   ▼
┌─────────────────┐
│   Manus.run()   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  状态: IDLE     │
│  添加用户消息    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 状态: RUNNING   │
└────────┬────────┘
         │
         ▼
    ┌─────────┐
    │ step()  │  ←──┐
    └────┬────┘    │
         │         │
    ┌────▼────┐    │
    │ think() │    │
    └────┬────┘    │
         │         │
    ┌────▼────┐    │
    │ 调用LLM │    │
    └────┬────┘    │
         │         │
    ┌────▼────┐    │
    │解析工具调用│   │
    └────┬────┘    │
         │         │
    ┌────▼────┐    │
    │  act()  │    │
    └────┬────┘    │
         │         │
    ┌────▼────┐    │
    │执行工具 │    │
    └────┬────┘    │
         │         │
    ┌────▼────┐    │
    │添加结果 │    │
    └────┬────┘    │
         │         │
    ┌────▼────┐    │
    │检查终止 │    │
    └────┬────┘    │
         │         │
    ┌────▼────┐    │
    │未完成？ │───┘
    └────┬────┘
         │
    ┌────▼────┐
    │状态:   │
    │FINISHED│
    └────────┘
```

### 4.2 工具调用流程

```
ToolCallAgent.think()
   │
   ▼
构建消息列表（历史对话 + 下一步提示）
   │
   ▼
调用 LLM.ask_tool()
   │
   ├─> 传入可用工具列表
   ├─> 传入工具选择模式（AUTO/REQUIRED/NONE）
   └─> 返回工具调用响应
   │
   ▼
解析工具调用
   │
   ├─> tool_calls: List[ToolCall]
   └─> 保存到 self.tool_calls
   │
   ▼
ToolCallAgent.act()
   │
   ▼
遍历 tool_calls
   │
   ├─> 查找工具（available_tools.find()）
   ├─> 解析参数（JSON）
   ├─> 执行工具（tool.execute()）
   ├─> 获取结果（ToolResult）
   └─> 添加到记忆（tool message）
   │
   ▼
检查终止条件
   │
   ├─> 如果调用了 Terminate 工具 → 完成
   └─> 否则 → 继续下一步
```

### 4.3 多智能体流程

```
PlanningFlow.execute()
   │
   ▼
使用 PlanningTool 分解任务
   │
   ├─> 输入：用户任务描述
   └─> 输出：计划步骤列表
   │
   ▼
遍历计划步骤
   │
   ├─> 获取步骤类型
   ├─> 选择执行者（get_executor()）
   ├─> 执行步骤（executor.run()）
   └─> 更新步骤状态
   │
   ▼
检查计划完成情况
   │
   ├─> 所有步骤完成 → 返回结果
   ├─> 有步骤阻塞 → 重新规划
   └─> 有步骤失败 → 重试或调整
```

### 4.4 MCP 工具集成流程

```
Manus.initialize_mcp_servers()
   │
   ▼
读取 MCP 配置（config/mcp.json）
   │
   ├─> SSE 服务器 → connect_sse()
   └─> STDIO 服务器 → connect_stdio()
   │
   ▼
建立连接
   │
   ├─> 初始化 MCP 客户端
   ├─> 获取服务器工具列表
   └─> 创建 MCPClientTool 实例
   │
   ▼
添加到可用工具
   │
   └─> available_tools.add_tools(*mcp_tools)
   │
   ▼
工具调用时
   │
   ├─> 识别 MCP 工具
   ├─> 调用 MCP 服务器
   └─> 返回结果
```

---

## 五、关键技术实现

### 5.1 异步执行机制

**核心**: 使用 `asyncio` 实现异步并发

**优势**:
- 非阻塞 I/O 操作
- 高效的并发处理
- 更好的资源利用

**示例**:

```python
# 异步工具执行
async def execute_tools(self, tool_calls: List[ToolCall]):
    """并发执行多个工具"""
    tasks = [
        self.execute_single_tool(tc) 
        for tc in tool_calls
    ]
    results = await asyncio.gather(*tasks)
    return results
```

### 5.2 Token 管理

**问题**: LLM API 有 Token 限制

**解决方案**:
- 精确计算 Token 数量
- 消息截断策略
- 错误处理和重试

**Token 计算**:

```python
def count_message_tokens(self, messages: List[dict]) -> int:
    """计算消息 Token 数"""
    total = 2  # 基础格式 Token
    
    for message in messages:
        total += 4  # 每条消息基础 Token
        
        # 角色 Token
        total += len(self.tokenizer.encode(message.get("role", "")))
        
        # 内容 Token
        content = message.get("content", "")
        if isinstance(content, str):
            total += self.count_text(content)
        elif isinstance(content, list):
            total += self.count_content(content)
        
        # 工具调用 Token
        if "tool_calls" in message:
            total += self.count_tool_calls(message["tool_calls"])
    
    return total
```

### 5.3 错误处理与重试

**策略**:
- 网络错误：指数退避重试
- Token 超限：截断消息或报错
- 工具执行失败：返回错误信息
- 超时：终止执行

**实现**:

```python
@retry(
    retry=retry_if_exception_type((RateLimitError, APIError)),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=10),
)
async def ask(self, messages: List[Message], ...):
    """带重试的 LLM 调用"""
    try:
        response = await self.client.chat.completions.create(...)
        return response
    except TokenLimitExceeded as e:
        # Token 超限处理
        raise
    except RateLimitError:
        # 速率限制，等待重试
        raise
```

### 5.4 卡死检测机制

**问题**: 智能体可能陷入循环

**解决方案**:

```python
def is_stuck(self) -> bool:
    """检测是否卡死"""
    if len(self.memory.messages) < 2:
        return False
    
    last_message = self.memory.messages[-1]
    if not last_message.content:
        return False
    
    # 检查最近是否有重复内容
    duplicate_count = sum(
        1
        for msg in reversed(self.memory.messages[:-1])
        if msg.role == "assistant" 
        and msg.content == last_message.content
    )
    
    return duplicate_count >= self.duplicate_threshold

def handle_stuck_state(self):
    """处理卡死状态"""
    stuck_prompt = (
        "Observed duplicate responses. "
        "Consider new strategies and avoid repeating ineffective paths."
    )
    self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
```

### 5.5 浏览器上下文管理

**挑战**: 浏览器操作需要上下文信息

**解决方案**:

```python
class BrowserContextHelper:
    """管理浏览器上下文"""
    
    async def format_next_step_prompt(self) -> str:
        """格式化包含浏览器上下文的提示词"""
        # 1. 获取当前页面截图
        screenshot = await self.browser.get_screenshot()
        
        # 2. 提取 DOM 结构
        dom = await self.browser.get_dom()
        
        # 3. 分析可交互元素
        elements = self.analyze_interactive_elements(dom)
        
        # 4. 生成上下文提示词
        context = f"""
        当前页面状态：
        - URL: {self.browser.current_url}
        - 可交互元素: {len(elements)} 个
        - 页面标题: {self.browser.title}
        
        建议操作：
        {self.suggest_actions(elements)}
        """
        
        return context
```

---

## 六、扩展开发指南

### 6.1 创建自定义工具

**步骤**:

1. **继承 BaseTool**:

```python
from app.tool.base import BaseTool, ToolResult

class MyCustomTool(BaseTool):
    name: str = "my_custom_tool"
    description: str = "我的自定义工具描述"
    parameters: dict = {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "参数1的描述",
            },
        },
        "required": ["param1"],
    }
    
    async def execute(self, param1: str) -> ToolResult:
        """执行工具逻辑"""
        try:
            # 执行操作
            result = do_something(param1)
            return self.success_response(result)
        except Exception as e:
            return self.fail_response(str(e))
```

2. **注册工具**:

```python
# 在 Manus 中添加工具
agent = Manus()
agent.available_tools.add_tools(MyCustomTool())
```

### 6.2 创建自定义智能体

**步骤**:

1. **继承 ToolCallAgent**:

```python
from app.agent.toolcall import ToolCallAgent
from app.tool import ToolCollection, MyCustomTool

class MyCustomAgent(ToolCallAgent):
    name: str = "MyCustomAgent"
    description: str = "我的自定义智能体"
    
    system_prompt: str = "你是一个专业的..."
    next_step_prompt: str = "下一步应该..."
    
    available_tools: ToolCollection = ToolCollection(
        MyCustomTool(),
        # 其他工具...
    )
    
    max_steps: int = 20
```

2. **使用智能体**:

```python
agent = MyCustomAgent()
await agent.run("执行任务")
```

### 6.3 集成 MCP 服务器

**步骤**:

1. **配置 MCP 服务器** (`config/mcp.json`):

```json
{
  "mcpServers": {
    "my_server": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "my_mcp_server"]
    }
  }
}
```

2. **自动加载**:

```python
# Manus 会自动加载配置的 MCP 服务器
agent = await Manus.create()
# MCP 工具已自动添加到 available_tools
```

### 6.4 创建自定义流程

**步骤**:

1. **继承 BaseFlow**:

```python
from app.flow.base import BaseFlow
from app.agent.base import BaseAgent

class MyCustomFlow(BaseFlow):
    async def execute(self, input_text: str) -> str:
        """执行自定义流程"""
        # 1. 解析输入
        # 2. 选择智能体
        # 3. 执行任务
        # 4. 返回结果
        pass
```

2. **注册流程**:

```python
from app.flow.flow_factory import FlowFactory, FlowType

# 在 flow_factory.py 中注册
FlowFactory.register_flow(FlowType.MY_CUSTOM, MyCustomFlow)
```

---

## 七、配置与部署

### 7.1 环境配置

**Python 版本**: 3.12+

**安装依赖**:

```bash
# 方式1: 使用 pip
pip install -r requirements.txt

# 方式2: 使用 uv（推荐）
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

**浏览器工具**:

```bash
playwright install
```

### 7.2 配置文件设置

**1. 创建配置文件**:

```bash
cp config/config.example.toml config/config.toml
```

**2. 配置 LLM**:

```toml
[llm]
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."
max_tokens = 4096
temperature = 0.0
api_type = "OpenAI"
```

**3. 配置 MCP** (可选):

```bash
cp config/mcp.example.json config/mcp.json
```

### 7.3 运行模式

**1. 单智能体模式**:

```bash
python main.py
# 或
python main.py --prompt "你的任务"
```

**2. 多智能体流程**:

```bash
python run_flow.py
```

**3. MCP 模式**:

```bash
python run_mcp.py
# 或交互模式
python run_mcp.py --interactive
```

### 7.4 Docker 部署

**Dockerfile**:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

**构建和运行**:

```bash
docker build -t openmanus .
docker run -v $(pwd)/config:/app/config openmanus
```

---

## 八、最佳实践

### 8.1 提示词设计

**原则**:
- 清晰明确的任务描述
- 提供足够的上下文信息
- 明确工具使用规则
- 设置合理的约束条件

**示例**:

```python
SYSTEM_PROMPT = """
你是一个专业的 AI 助手，可以：
1. 执行 Python 代码
2. 浏览网页
3. 搜索信息
4. 编辑文件

工作目录: {directory}

重要规则：
- 在执行代码前，先分析需求
- 使用浏览器工具时，先提取页面信息
- 文件操作要谨慎，避免删除重要文件
"""
```

### 8.2 工具选择策略

**原则**:
- 工具描述要准确
- 参数定义要完整
- 错误处理要完善
- 返回结果要结构化

**示例**:

```python
class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = (
        "执行特定操作。"
        "适用于场景A、场景B。"
        "不适用于场景C。"
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "输入参数，必须是有效的...",
            },
        },
        "required": ["input"],
    }
```

### 8.3 错误处理

**原则**:
- 捕获所有可能的异常
- 返回有意义的错误信息
- 记录错误日志
- 提供恢复建议

**示例**:

```python
async def execute(self, **kwargs) -> ToolResult:
    try:
        result = await self._do_work(**kwargs)
        return self.success_response(result)
    except ValueError as e:
        logger.error(f"参数错误: {e}")
        return self.fail_response(f"参数错误: {e}")
    except TimeoutError:
        logger.error("操作超时")
        return self.fail_response("操作超时，请重试")
    except Exception as e:
        logger.exception("未知错误")
        return self.fail_response(f"执行失败: {str(e)}")
```

### 8.4 性能优化

**建议**:
- 使用异步操作
- 并发执行独立任务
- 缓存重复计算结果
- 限制 Token 使用

**示例**:

```python
# 并发执行多个工具
async def execute_multiple_tools(self, tool_calls: List[ToolCall]):
    tasks = [self.execute_tool(tc) for tc in tool_calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

### 8.5 安全考虑

**建议**:
- 代码执行使用沙箱
- 限制文件系统访问
- 验证用户输入
- 限制网络访问

**示例**:

```python
# 使用沙箱执行代码
[sandbox]
use_sandbox = true
image = "python:3.12-slim"
network_enabled = false
memory_limit = "512m"
cpu_limit = 1.0
```

---

## 九、常见问题

### 9.1 Token 超限

**问题**: 上下文太长导致 Token 超限

**解决方案**:
- 减少历史消息数量
- 使用消息摘要
- 截断长文本
- 使用更大的模型

### 9.2 工具执行失败

**问题**: 工具执行出错

**解决方案**:
- 检查工具参数
- 查看错误日志
- 验证环境配置
- 添加重试机制

### 9.3 浏览器操作失败

**问题**: 浏览器操作不成功

**解决方案**:
- 检查元素选择器
- 等待页面加载
- 使用截图调试
- 检查网络连接

### 9.4 MCP 连接失败

**问题**: MCP 服务器连接失败

**解决方案**:
- 检查服务器配置
- 验证服务器运行状态
- 检查网络连接
- 查看日志信息

---

## 十、总结

OpenManus 是一个功能完整、设计良好的 AI 智能体框架。通过模块化的架构设计、丰富的工具生态和灵活的扩展机制，它能够支持各种复杂的任务场景。

**核心优势**:
- ✅ 清晰的架构设计
- ✅ 丰富的工具支持
- ✅ 灵活的扩展机制
- ✅ 完善的错误处理
- ✅ 良好的文档支持

**适用场景**:
- 自动化任务执行
- 数据分析与可视化
- 网页交互与信息提取
- 代码生成与执行
- 多智能体协作

**未来发展方向**:
- 更多工具集成
- 更强的规划能力
- 更好的错误恢复
- 更完善的文档

---

**文档版本**: 1.0  
**最后更新**: 2025-03-07  
**维护者**: OpenManus Team

