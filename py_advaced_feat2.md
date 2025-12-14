## 五、装饰器 (Decorators)

### 5.1 什么是装饰器？

装饰器是一种语法糖，用于修改或扩展函数/方法的功能。

### 5.2 基础装饰器

```python
# 定义装饰器
def my_decorator(func):
    def wrapper():
        print("执行前")
        func()
        print("执行后")
    return wrapper

# 使用装饰器
@my_decorator
def say_hello():
    print("Hello")

say_hello()
# 输出：
# 执行前
# Hello
# 执行后
```

### 5.3 项目中常用的装饰器

#### 5.3.1 @property - 属性装饰器

```python
class Person:
    def __init__(self, name):
        self._name = name  # 私有属性（约定用 _ 开头）
    
    @property
    def name(self):
        """获取名字（像属性一样使用）"""
        return self._name
    
    @name.setter
    def name(self, value):
        """设置名字"""
        self._name = value

# 使用
person = Person("Alice")
print(person.name)      # "Alice"（调用 getter）
person.name = "Bob"     # 调用 setter
```

**项目示例**：
```python
# 来自 app/agent/base.py
@property
def messages(self) -> List[Message]:
    """获取消息列表"""
    return self.memory.messages

@messages.setter
def messages(self, value: List[Message]):
    """设置消息列表"""
    self.memory.messages = value

# 使用
agent.messages = [msg1, msg2]  # 调用 setter
msgs = agent.messages          # 调用 getter
```

#### 5.3.2 @classmethod - 类方法装饰器

```python
class Message:
    role: str
    content: str
    
    def __init__(self, role, content):
        self.role = role
        self.content = content
    
    @classmethod
    def user_message(cls, content: str):
        """类方法：通过类调用，返回实例"""
        return cls(role="user", content=content)
    
    @classmethod
    def system_message(cls, content: str):
        return cls(role="system", content=content)

# 使用
msg1 = Message.user_message("Hello")    # 通过类调用
msg2 = Message.system_message("System")  # 通过类调用
```

**项目示例**：
```python
# 来自 app/schema.py
@classmethod
def user_message(cls, content: str) -> "Message":
    """创建用户消息"""
    return cls(role=Role.USER, content=content)

# 使用
msg = Message.user_message("Hello")
```

#### 5.3.3 @staticmethod - 静态方法装饰器

```python
class MathUtils:
    @staticmethod
    def add(a, b):
        """静态方法：不需要访问类或实例"""
        return a + b

# 使用
result = MathUtils.add(1, 2)  # 3
```

#### 5.3.4 @abstractmethod - 抽象方法装饰器

```python
from abc import ABC, abstractmethod

class BaseTool(ABC):
    @abstractmethod
    async def execute(self, **kwargs):
        """子类必须实现"""
        pass
```

#### 5.3.5 @model_validator - Pydantic 验证器

```python
from pydantic import BaseModel, model_validator

class Agent(BaseModel):
    name: str
    max_steps: int = 10
    
    @model_validator(mode="after")
    def initialize_agent(self):
        """对象创建后自动调用"""
        if self.max_steps < 1:
            self.max_steps = 1
        return self

# 创建对象时自动调用
agent = Agent(name="Test", max_steps=0)
print(agent.max_steps)  # 1（被验证器修改）
```

**项目示例**：
```python
# 来自 app/agent/base.py
@model_validator(mode="after")
def initialize_agent(self) -> "BaseAgent":
    """初始化智能体"""
    if self.llm is None:
        self.llm = LLM()
    return self
```

#### 5.3.6 @retry - 重试装饰器

```python
from tenacity import retry, stop_after_attempt, wait_random_exponential

@retry(
    stop=stop_after_attempt(3),  # 最多重试 3 次
    wait=wait_random_exponential(multiplier=1, max=10)  # 指数退避
)
async def fetch_data():
    """如果失败会自动重试"""
    # 网络请求代码
    pass
```

**理解要点**：
- `@property` - 将方法变成属性
- `@classmethod` - 类方法，通过类调用
- `@staticmethod` - 静态方法，不需要实例
- `@abstractmethod` - 抽象方法，必须实现
- `@model_validator` - Pydantic 验证器
- `@retry` - 自动重试装饰器

---

## 六、上下文管理器 (Context Managers)

### 6.1 什么是上下文管理器？

上下文管理器用于管理资源的获取和释放，**确保资源被正确清理**。

### 6.2 基础用法（with 语句）

```python
# 文件操作（自动关闭文件）
with open("file.txt", "r") as f:
    content = f.read()
# 文件自动关闭

# 等价于
f = open("file.txt", "r")
try:
    content = f.read()
finally:
    f.close()
```

### 6.3 自定义上下文管理器

```python
class MyContext:
    def __enter__(self):
        print("进入")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("退出")
        return False

# 使用
with MyContext():
    print("执行代码")
# 输出：
# 进入
# 执行代码
# 退出
```

### 6.4 异步上下文管理器

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def my_async_context():
    print("进入")
    try:
        yield  # 返回控制权
    finally:
        print("退出")

# 使用
async def main():
    async with my_async_context():
        print("执行代码")

asyncio.run(main())
```

**项目示例**：
```python
# 来自 app/agent/base.py
@asynccontextmanager
async def state_context(self, new_state: AgentState):
    """状态上下文管理器"""
    previous_state = self.state
    self.state = new_state
    try:
        yield  # 执行代码
    except Exception as e:
        self.state = AgentState.ERROR
        raise e
    finally:
        self.state = previous_state  # 恢复原状态

# 使用
async with self.state_context(AgentState.RUNNING):
    # 执行代码，状态自动管理
    await self.step()
```

**理解要点**：
- `with` 语句自动管理资源
- `__enter__` 和 `__exit__` 定义同步上下文管理器
- `@asynccontextmanager` 定义异步上下文管理器
- `yield` 返回控制权

---

## 七、枚举 (Enum)

### 7.1 什么是枚举？

枚举用于定义一组固定的常量值。

### 7.2 基础用法

```python
from enum import Enum

class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

# 使用
color = Color.RED
print(color)        # Color.RED
print(color.value)  # "red"
print(color.name)   # "RED"
```

### 7.3 项目中的实际例子

```python
# 来自 app/schema.py
class AgentState(str, Enum):  # 继承 str，值也是字符串
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"

# 使用
state = AgentState.IDLE
print(state)        # AgentState.IDLE
print(state.value)  # "IDLE"

# 比较
if state == AgentState.IDLE:
    print("空闲状态")
```

**理解要点**：
- `Enum` 定义枚举类
- `str, Enum` 表示值也是字符串类型
- 枚举值通过 `.value` 访问
- 枚举名通过 `.name` 访问

---

## 八、泛型 (Generics)

### 8.1 什么是泛型？

泛型允许定义可以处理多种类型的类或函数。

### 8.2 基础用法

```python
from typing import TypeVar, Generic, List

# 定义类型变量
T = TypeVar('T')

class Box(Generic[T]):
    def __init__(self, item: T):
        self.item = item
    
    def get(self) -> T:
        return self.item

# 使用
box_int = Box[int](10)      # 整数盒子
box_str = Box[str]("hello")  # 字符串盒子
```

### 8.3 项目中的实际例子

```python
# 来自 app/tool/browser_use_tool.py
from typing import Generic, TypeVar

Context = TypeVar("Context")

class BrowserUseTool(BaseTool, Generic[Context]):
    """泛型工具类"""
    pass
```

**理解要点**：
- `TypeVar` 定义类型变量
- `Generic[T]` 表示泛型类
- 可以在使用时指定具体类型

---

## 九、魔法方法 (Magic Methods)

### 9.1 什么是魔法方法？

魔法方法是以双下划线开头和结尾的特殊方法，用于定义类的行为。

### 9.2 常用魔法方法

```python
class MyClass:
    def __init__(self, value):
        """初始化方法"""
        self.value = value
    
    def __str__(self):
        """字符串表示（print 时调用）"""
        return f"MyClass({self.value})"
    
    def __repr__(self):
        """对象表示（调试时调用）"""
        return f"MyClass(value={self.value})"
    
    def __add__(self, other):
        """加法运算符 +"""
        return MyClass(self.value + other.value)
    
    def __len__(self):
        """len() 函数"""
        return len(self.value)

# 使用
obj = MyClass(10)
print(obj)        # MyClass(10)（调用 __str__）
obj2 = MyClass(20)
obj3 = obj + obj2  # MyClass(30)（调用 __add__）
```

### 9.3 项目中的实际例子

```python
# 来自 app/schema.py
class Message(BaseModel):
    def __add__(self, other) -> List["Message"]:
        """支持 Message + list 或 Message + Message"""
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError("不支持的类型")
    
    def __radd__(self, other) -> List["Message"]:
        """支持 list + Message"""
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError("不支持的类型")

# 使用
msg1 = Message.user_message("Hello")
msg2 = Message.user_message("World")
msgs = msg1 + msg2  # [msg1, msg2]（调用 __add__）
msgs = [msg1] + msg2  # [msg1, msg2]（调用 __radd__）
```

**理解要点**：
- `__init__` - 初始化
- `__str__` - 字符串表示
- `__add__` - 加法运算符
- `__radd__` - 右加法运算符
- `__len__` - len() 函数

---

## 十、单例模式

### 10.1 什么是单例模式？

单例模式确保一个类只有一个实例。

### 10.2 实现方式

```python
class Singleton:
    _instance = None
    
    def __new__(cls):
        """创建实例时调用"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# 使用
obj1 = Singleton()
obj2 = Singleton()
print(obj1 is obj2)  # True（同一个实例）
```

### 10.3 项目中的实际例子

```python
# 来自 app/config.py
class Config:
    _instance = None
    _lock = threading.Lock()  # 线程锁（多线程安全）
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:  # 线程安全
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

# 使用
config1 = Config()
config2 = Config()
print(config1 is config2)  # True（同一个实例）
```

**理解要点**：
- `__new__` 在 `__init__` 之前调用
- 使用类变量存储实例
- 使用锁保证线程安全

---

## 十一、其他高级特性

### 11.1 解包操作符

```python
# * 解包列表/元组
numbers = [1, 2, 3]
print(*numbers)  # 1 2 3

# ** 解包字典
params = {"name": "Alice", "age": 25}
func(**params)  # 等价于 func(name="Alice", age=25)

# 项目示例
kwargs = {"base64_image": image, **other_kwargs}  # 合并字典
```

### 11.2 列表推导式

```python
# 基础
numbers = [x * 2 for x in range(5)]  # [0, 2, 4, 6, 8]

# 带条件
evens = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]

# 项目示例
new_tools = [
    tool for tool in self.mcp_clients.tools 
    if tool.server_id == server_id
]
```

### 11.3 生成器表达式

```python
# 生成器（惰性计算）
gen = (x * 2 for x in range(5))
for value in gen:
    print(value)

# 项目示例
duplicate_count = sum(
    1
    for msg in reversed(self.memory.messages[:-1])
    if msg.role == "assistant" and msg.content == last_message.content
)
```

### 11.4 lambda 函数

```python
# 匿名函数
add = lambda x, y: x + y
print(add(1, 2))  # 3

# 项目示例
message_map = {
    "user": Message.user_message,
    "tool": lambda content, **kw: Message.tool_message(content, **kw),
}
```

### 11.5 f-string 格式化

```python
name = "Alice"
age = 25
message = f"Hello, {name}! You are {age} years old."
# "Hello, Alice! You are 25 years old."

# 项目示例
logger.info(f"Executing step {self.current_step}/{self.max_steps}")
```

### 11.6 Path 对象（路径操作）

```python
from pathlib import Path

# 创建路径对象
path = Path("config/config.toml")

# 检查存在
if path.exists():
    print("文件存在")

# 读取文件
content = path.read_text()

# 项目示例
config_path = PROJECT_ROOT / "config" / "config.toml"
```

### 11.7 默认工厂函数

```python
from typing import List

# 错误：可变默认参数
def bad_func(items=[]):  # 危险！
    items.append(1)
    return items

# 正确：使用 None 和默认值
def good_func(items=None):
    if items is None:
        items = []
    items.append(1)
    return items

# 更好：使用 default_factory
from pydantic import Field

class MyModel(BaseModel):
    items: List[str] = Field(default_factory=list)  # 每次创建新列表
```

### 11.8 类型忽略注释

```python
# 告诉类型检查器忽略这行
role: ROLE_TYPE = Field(...)  # type: ignore

# 当类型检查器无法正确推断类型时使用
```

---

## 十二、快速参考表

| 特性 | 语法 | 用途 | 示例 |
|------|------|------|------|
| 类型提示 | `name: str` | 标注类型 | `def func(x: int) -> str:` |
| Optional | `Optional[str]` | 可选类型 | `name: Optional[str] = None` |
| Union | `Union[str, int]` | 多类型 | `value: Union[str, int]` |
| List | `List[str]` | 列表类型 | `items: List[str]` |
| Dict | `Dict[str, int]` | 字典类型 | `scores: Dict[str, int]` |
| async/await | `async def`, `await` | 异步编程 | `async def func(): await task()` |
| @property | `@property` | 属性访问 | `@property def name(self):` |
| @classmethod | `@classmethod` | 类方法 | `@classmethod def create(cls):` |
| @abstractmethod | `@abstractmethod` | 抽象方法 | `@abstractmethod def step(self):` |
| Enum | `class Color(Enum):` | 枚举 | `class State(Enum): IDLE = "idle"` |
| BaseModel | `class User(BaseModel):` | 数据模型 | Pydantic 模型 |
| Field | `Field(...)` | 字段定义 | `name: str = Field(...)` |
| with | `with open():` | 上下文管理 | `with context():` |
| __init__ | `def __init__(self):` | 初始化 | 构造函数 |
| __add__ | `def __add__(self, other):` | 加法运算符 | `obj1 + obj2` |
| **kwargs | `**kwargs` | 关键字参数 | `def func(**kwargs):` |
| *args | `*args` | 可变参数 | `def func(*args):` |

---

## 十三、学习建议

### 13.1 循序渐进

1. **先理解基础概念**
   - 类型提示：理解 `str`, `int`, `Optional` 等
   - 异步编程：理解 `async/await` 的基本用法

2. **再学习高级特性**
   - Pydantic：理解数据验证
   - 装饰器：理解 `@property`, `@classmethod` 等
   - 抽象基类：理解继承和抽象方法

3. **最后学习设计模式**
   - 单例模式
   - 上下文管理器
   - 工厂模式

### 13.2 实践建议

1. **阅读代码时**：
   - 遇到不认识的语法，先查文档
   - 理解每个特性的用途
   - 尝试自己写简单的例子

2. **编写代码时**：
   - 先写基础版本
   - 逐步添加类型提示
   - 使用 Pydantic 验证数据
   - 添加异步支持

3. **调试代码时**：
   - 使用类型检查工具（mypy）
   - 查看错误信息
   - 理解异常原因

### 13.3 推荐资源

- **Python 官方文档**：https://docs.python.org/3/
- **Pydantic 文档**：https://docs.pydantic.dev/
- **asyncio 文档**：https://docs.python.org/3/library/asyncio.html
- **typing 模块**：https://docs.python.org/3/library/typing.html

---

## 十四、常见问题解答

### Q1: 类型提示会影响程序运行吗？

**A**: 不会。类型提示只是用于代码检查和 IDE 提示，运行时会被忽略。

### Q2: 什么时候使用 async/await？

**A**: 当需要执行 I/O 操作（网络请求、文件读写、数据库查询）时使用异步，可以提高性能。

### Q3: @property 和普通方法有什么区别？

**A**: `@property` 让方法可以像属性一样访问（不需要括号），通常用于计算属性或访问私有属性。

### Q4: 为什么要使用 Pydantic？

**A**: Pydantic 提供自动数据验证、类型转换、文档生成等功能，让代码更安全、更易维护。

### Q5: 抽象基类有什么用？

**A**: 抽象基类定义接口，确保子类实现必要的方法，提高代码的可维护性和可扩展性。

### Q6: 单例模式什么时候使用？

**A**: 当需要确保全局只有一个实例时使用，比如配置类、数据库连接等。

---

**文档版本**: 1.0  
**最后更新**: 2025-03-07  
**适用对象**: Python 基础学习者

