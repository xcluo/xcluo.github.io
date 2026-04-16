---
title: LangChain
---

`pip install longchain`


- https://python.langchain.com/docs/how_to/

### 数据处理
#### [Document Loader](document_loader.md)
1. PDF files
2. web pages
3. CSV data
4. data from a directory
5. HTML data
6. JSON data
7. Markdown data
8. Microsoft Office Data
9. customized document loader
#### Text Spliter

### Embedding相关
```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

vector_store = InMemoryVectorStore.from_documents(pages, OpenAIEmbeddings())
docs = vector_store.similarity_search("What is LayoutParser?", k=2)
for doc in docs:
    print(f'Page {doc.metadata["page"]}: {doc.page_content[:300]}\n')
```

### [Agent](https://docs.langchain.com/oss/python/langchain/overview)
要求Python 3.10+
```
pip install -U langchain
pip install -U langchain-openai
pip install -U langchain-anthropic
pip install langchain-community
```

#### BaseChatModel
调用的llm最好为能将任务细分的推理模型

1. `ChatOpenAI`
    ```python
    from langchain_openai import ChatOpenAI

    # 自定义LLM API Model # 
    # static model，运行时保持不变
    llm = ChatOpenAI(
        base_url=Config.BASE_URL,
        api_key=Config.API_KEY,
        model=Config.MODEL_NAME,
        temperature=0.1,
        max_tokens=1000,
        timeout=30,
        max_retries=2,
    )

    # 输入input，调用大模型进行对话获得完整结果
    llm.invoke(
        input,
    ) -> AIMessage: 
    # 输入input，调用大模型进行对话获得流式结果
    llm.stream(
       input,
    ) -> AIMessage: 
    ```

2. BaseChatModel
    ```python
    from langchain.chat_models import init_chat_model
    llm = init_chat_model(
        model: str | None = None,
        *,
        model_provider: str | None = None,
    )
    ```

#### tools

- `@tool` 将普通函数转化为智能体可调用的工具，自动处理参数解析、描述注册等  
- `ToolRuntime` LangChain 工具运行时的上下文类，用于传递工具执行时额外的上下文（如用户 ID、请求 ID、权限信息等），支持泛型`StateT`和`ContextT`，常用属性为`state: StateT, content: ContextT`
- `@dataclass` 快速定义简单的数据类，替代手动写 `__init__`，等价于继承`pydantic.BaseModel`

```python
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"
```

#### system_prompt

```python
system_prompt: str | SystemMessage | None = None
```

#### middleware
- `@wrap_model_call`将普通函数转化为「智能体中间件」，使其能拦截智能体的模型调用请求；保证函数遵循 “接收 ModelRequest → 处理 → 调用 handler → 返回 ModelResponse” 的中间件规范；
- `ModelRequest` 常用参数为`model: BaseChatModel`，`messages: list[AnyMessage]`，`tool_choice: Any | None`以及`tools: list[BaseTool | dict]` 等

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

# dynamic models，根据状态和文本动态选择model
basic_model = ChatOpenAI(model="gpt-4o-mini")
advanced_model = ChatOpenAI(model="gpt-4o")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model,  # Default model
    tools=tools,
    middleware=[dynamic_model_selection]
)
```

- `@wrap_tool_call` 定义了一个工具调用异常兜底中间件，若出现调用错误，将执行【中间件捕获异常 → 返回自定义错误消息 → 大模型基于错误消息继续交互（而非崩溃）】

```python
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="gpt-4o",
    tools=[search, get_weather],
    middleware=[handle_tool_errors]
)
```

- `@dynamic_prompt` LangChain 会自动将该函数的返回值作为智能体的「系统提示词」
```python
from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest


class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt

agent = create_agent(
    model="gpt-4o",
    tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context
)

# The system prompt will be set dynamically based on context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain machine learning"}]},
    context={"user_role": "expert"}
)
```

#### Add Memory
[Add and manage memory](https://docs.langchain.com/oss/python/langgraph/add-memory#manage-short-term-memory)
- 增加用户偏好
```python
checkpointer = InMemorySaver()
# In production, use a persistent checkpointer that saves to a database

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from typing import Any


class CustomState(AgentState):
    user_preferences: dict

class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState
    tools = [tool1, tool2]

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        ...

agent = create_agent(
    model,
    tools=tools,
    middleware=[CustomMiddleware()]
)

# The agent can now track additional state beyond messages
result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})


from langchain.agents import AgentState


class CustomState(AgentState):
    user_preferences: dict

agent = create_agent(
    model,
    tools=[tool1, tool2],
    state_schema=CustomState
)
# The agent can now track additional state beyond messages
result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})
```

#### response_format
新增结构性返回结果`structured_response`

- `ToolStrategy` 约束工具调用的输出格式的类，支持泛型`SchemaT`，核心作用是告诉大模型必须按照`SchemaT`结构返回结果（会自动校验输出，若输出不符合会自动重试或出发报错）
```python
from langchain.agents.structured_output import ToolStrategy

@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None

# Agent新增结构性返回结果
response_format=ToolStrategy(ResponseFormat)
```


#### Agent/CompiledStateGraph

1. `create_agent` 创建ReAct Agent
    ```python
    from langchain.agents import create_agent
    from langgraph.prebuilt import create_react_agent

    from langgraph.checkpoint.memory import InMemorySaver

    agent = create_agent(
        # model="claude-sonnet-4-5-20250929" 直接使用claude
        model=llm,
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
        context_schema=Context,
        response_format=ToolStrategy(ResponseFormat),
        checkpointer=checkpointer
    )

    # 启动agent任务
    res = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        context=Context(user_id="1")
    )

    print(res['structured_response'])
    ```
2. `create_react_agent`



- BaseMessage
- SystemMessage
- HumanMessage
- AIMessage
- ToolMessage