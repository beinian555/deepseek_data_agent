from __future__ import annotations

import base64
import os
import sqlite3
from collections.abc import Callable
from typing import Annotated

import streamlit as st
from e2b_code_interpreter import Sandbox
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict


def _build_python_interpreter_tool(*, e2b_api_key: str):
    @tool
    def python_interpreter(code: str):
        """
        [E2B 云端沙盒版] Python 代码执行器。
        代码将在安全的云端隔离环境中运行，支持 pandas, matplotlib 等库。
        """

        if not e2b_api_key:
            return "❌ 错误: 未配置 E2B API Key，无法启动云端沙盒。"

        try:
            with Sandbox(api_key=e2b_api_key) as sbx:
                if "current_file_path" in st.session_state:
                    local_path = st.session_state["current_file_path"]
                    if os.path.exists(local_path):
                        with open(local_path, "rb") as f:
                            sbx.files.write(os.path.basename(local_path), f)

                clean_code = code.strip().replace("```python", "").replace("```", "")

                header: list[str] = [
                    "import pandas as pd",
                    "import matplotlib.pyplot as plt",
                    "plt.switch_backend('Agg')",
                ]

                if "current_file_path" in st.session_state:
                    fname = os.path.basename(st.session_state["current_file_path"])
                    if fname.endswith(".csv"):
                        header.append(f"df = pd.read_csv('{fname}')")
                    elif fname.endswith((".xls", ".xlsx")):
                        header.append(f"df = pd.read_excel('{fname}')")

                full_code = "\n".join(header) + "\n" + clean_code
                execution = sbx.run_code(full_code)

                if execution.error:
                    return (
                        "❌ 代码执行报错:\n"
                        f"{execution.error.name}: {execution.error.value}\n"
                        f"{execution.error.traceback}"
                    )

                output_log = ""
                if execution.logs.stdout:
                    output_log += f"📄 输出:\n{str(execution.logs.stdout)}\n"

                for result in execution.results:
                    if hasattr(result, "png") and result.png:
                        img_data = base64.b64decode(result.png)
                        with open("result.png", "wb") as f:
                            f.write(img_data)
                        output_log += "\n🖼️ 图表已生成并保存为 result.png"

                return output_log or "代码执行成功，无文本输出。"

        except Exception as e:
            return f"沙盒连接或执行失败: {e}"

    return python_interpreter


def _build_lookup_policy_tool():
    @tool
    def lookup_policy(query: str):
        """
        只有当用户询问具体的业务文档、政策、报告内容时才使用此工具。
        输入应该是具体的查询问题。
        """

        if "vector_db" not in st.session_state:
            return "用户还未上传文档，无法检索。"

        db = st.session_state["vector_db"]
        retriever = db.as_retriever(search_kwargs={"k": 3})
        results = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in results])

    return lookup_policy


@st.cache_resource
def build_agent(
    api_key: str,
    *,
    e2b_api_key: str,
    data_context_provider: Callable[[], tuple[str, str]],
):
    tools = [
        _build_python_interpreter_tool(e2b_api_key=e2b_api_key),
        _build_lookup_policy_tool(),
    ]

    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    def agent_node(state: AgentState):
        messages = state["messages"]

        file_path, data_context = data_context_provider()
        sys_msg = SystemMessage(
            content=f"""你是一个高级数据分析智能体，拥有两大核心能力：
1. 🐍 **Python 代码执行**：用于处理数据、计算统计量、绘制图表。
2. 📚 **知识库检索**：用于查询具体的业务文档、政策、报告原文。

【当前数据环境】
1. 数据文件路径: '{file_path}'
2. 数据摘要: {data_context}

【调度决策准则】
- 📊 **遇到数据计算、画图需求**：请编写 Python 代码，使用 `df` 变量。
- ❓ **遇到业务含义、政策解释、背景知识查询**：请务必调用 `lookup_policy` 工具检索知识库，**严禁凭空编造**。
- 🤝 **混合需求**：如果用户问“根据最新的销售政策（PDF），分析这份数据（CSV）”，你需要先查知识库，理解政策，再写代码分析数据。

【云端执行须知】
1. 你的代码是在云端 Linux 沙盒中运行的。
2. 数据文件已经自动上传到当前目录。
3. ⚠️ 绘图时，为了避免 Linux 字体缺失导致乱码，**图表的标题和轴标签请尽量使用英文**。
   (例如: 使用 'Sales' 而不是 '销售额')

【执行铁律】
1. 🚫 严禁 input() 和 plt.show()。
2. ✅ 绘图必须保存为 'result.png'。
3. ✅ 查知识库时，如果查不到内容，请直接告诉用户“知识库中未找到相关信息”。
"""
        )

        new_messages = list(messages)
        if new_messages and isinstance(new_messages[0], SystemMessage):
            new_messages[0] = sys_msg
        else:
            new_messages.insert(0, sys_msg)

        return {"messages": [llm_with_tools.invoke(new_messages)]}

    tool_node = ToolNode(tools)
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    conn = sqlite3.connect("memory.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    return workflow.compile(checkpointer=memory)

