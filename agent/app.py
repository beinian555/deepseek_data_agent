import hashlib
import os
import sqlite3
import pickle
from pathlib import Path
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

import base64
import requests
from e2b_code_interpreter import Sandbox

from di_agent.utils import get_dataframe_info, save_uploaded_file

def rerank_with_hosted_api(
    query: str,
    docs: list[str],
    api_key: str,
    base_url: str,
    model: str,
    provider: str,
):
    """
    调用托管 Rerank API（支持 Jina / Cohere）并返回重排分数列表。
    返回长度与 docs 一致的 float 列表；失败时抛异常，由上层决定是否回退。
    """
    if not docs:
        return []

    provider = provider.lower().strip()
    headers = {"Content-Type": "application/json"}

    if provider == "cohere":
        url = base_url.rstrip("/") + "/v2/rerank"
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "query": query,
            "documents": docs,
            "top_n": len(docs),
        }
    else:
        # 默认按 Jina 协议
        url = base_url.rstrip("/") + "/rerank"
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "query": query,
            "documents": docs,
            "top_n": len(docs),
        }

    response = requests.post(url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()

    # 兼容常见返回结构:
    # {"results":[{"index":0,"relevance_score":0.9}, ...]}
    # {"data":[{"index":0,"score":0.9}, ...]}
    items = data.get("results") or data.get("data") or []
    if not items:
        raise ValueError("Rerank API 返回为空")

    scores = [0.0] * len(docs)
    for item in items:
        idx = item.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(docs):
            continue
        score = item.get("relevance_score", item.get("score", 0.0))
        scores[idx] = float(score)
    return scores


def _doc_dedupe_key(doc) -> str:
    return hashlib.sha256(doc.page_content.encode("utf-8", errors="ignore")).hexdigest()


def rrf_merge_doc_lists(
    *ranked_lists: list,
    rrf_k: int = 60,
) -> list:
    """
    Reciprocal Rank Fusion：多路有序结果合并为单一排序列表。
    ranked_lists 中每个元素为 Document 列表，按相关性从高到低。
    """
    scores: dict[str, float] = {}
    doc_by_key: dict[str, object] = {}
    for ranked in ranked_lists:
        if not ranked:
            continue
        for rank, doc in enumerate(ranked, start=1):
            key = _doc_dedupe_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            doc_by_key[key] = doc
    if not scores:
        return []
    ordered_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return [doc_by_key[k] for k in ordered_keys]


# --- 新增函数：处理 PDF 并构建向量库 + BM25（混合检索） ---
@st.cache_resource
def create_vector_db(file_path, api_key, base_url):
    project_root = Path(__file__).resolve().parent
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
        openai_api_base=base_url,
    )
    # 使用固定规则生成 persist 目录，避免不同 PDF/Key 的向量互相覆盖
    persist_hash = hashlib.sha256((file_path + api_key + base_url).encode("utf-8", errors="ignore")).hexdigest()[:16]
    persist_dir = str(project_root / ".chroma" / persist_hash)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="pdfqa",
    )
    # k 在检索时按侧边栏动态设置；但 BM25 的向量化结果（tokenize + BM25Okapi）做磁盘缓存
    bm25_dir = project_root / ".bm25" / persist_hash
    bm25_dir.mkdir(parents=True, exist_ok=True)
    bm25_pkl = bm25_dir / "bm25.pkl"
    try:
        if bm25_pkl.exists():
            with open(bm25_pkl, "rb") as f:
                payload = pickle.load(f)
            bm25_retriever = BM25Retriever(
                vectorizer=payload["vectorizer"],
                docs=payload["docs"],
                k=50,
            )
        else:
            bm25_retriever = BM25Retriever.from_documents(splits, k=50)
            with open(bm25_pkl, "wb") as f:
                pickle.dump({"vectorizer": bm25_retriever.vectorizer, "docs": bm25_retriever.docs}, f)
    except Exception:
        # 回退：如果 pickle 反序列化失败，则重新构建 BM25
        bm25_retriever = BM25Retriever.from_documents(splits, k=50)
    return vectorstore, bm25_retriever

st.set_page_config(page_title="DeepSeek 数据分析师", layout="wide")
st.title("DeepSeek 数据分析师")

with st.sidebar:
    st.header("⚙️ 控制台")
    api_key = st.text_input("DeepSeek API Key (用于对话)", type="password", value="")
    e2b_api_key = st.text_input(
        "E2B API Key (用于沙盒安全执行)", 
        type="password", 
        help="去 e2b.dev 申请，用于在云端隔离环境中运行代码"
    )
    embedding_api_key = st.text_input(
        "Embedding API Key (用于知识库)", 
        type="password", 
        value="", # 你的 Embedding 专用 Key
        help="用于 OpenAIEmbeddings 的 Key，通常是 OpenAI 官方或中转服务的 Key"
    )
    
    embedding_base_url = st.text_input(
        "Embedding Base URL", 
        value="https://poloai.top/v1", 
        help="如果是中转服务，请填写 Base URL"
    )

    st.divider()
    st.subheader("💾 记忆管理")
    thread_id = st.text_input("会话 ID (Thread ID)", value="session_1", help="输入不同的 ID 可以切换不同的对话历史")
    
    st.subheader("📂 数据加载")
    uploaded_file = st.file_uploader("上传数据表(CSV/Excel)", type=["csv", "xlsx"])
    uploaded_pdf = st.file_uploader("上传知识库 (PDF)", type=["pdf"])

    st.subheader("🔎 检索参数")
    st.session_state["rag_hybrid"] = st.checkbox(
        "启用混合检索 (BM25 + 向量)",
        value=st.session_state.get("rag_hybrid", True),
        help="两路各取 k 条，经 RRF 融合后再（可选）托管 Rerank，最后取 top_n。",
    )
    st.session_state["rag_rrf_k"] = st.slider(
        "RRF 平滑参数 k",
        min_value=30,
        max_value=90,
        value=int(st.session_state.get("rag_rrf_k", 60)),
        step=5,
        help="Reciprocal Rank Fusion 中的 k，常用 60。",
    )
    st.session_state["rag_candidates_k"] = st.slider(
        "每路召回数 (k)",
        min_value=10,
        max_value=200,
        value=int(st.session_state.get("rag_candidates_k", 50)),
        step=10,
        help="向量一路与 BM25 一路各取 k 条；关闭混合时仅向量一路。",
    )
    st.session_state["rag_top_n"] = st.slider(
        "最终使用片段数 (top_n)",
        min_value=3,
        max_value=12,
        value=int(st.session_state.get("rag_top_n", 6)),
        step=1,
        help="重排后取前 top_n 段拼进上下文，过大会增加 token 成本。",
    )
    st.subheader("🏷️ Rerank（托管 API）")
    st.session_state["rerank_provider"] = st.selectbox(
        "Rerank Provider",
        options=["jina", "cohere"],
        index=0 if st.session_state.get("rerank_provider", "jina") == "jina" else 1,
        help="选择托管重排服务商。Jina 与 Cohere 均支持。",
    )
    st.session_state["rerank_api_key"] = st.text_input(
        "Rerank API Key",
        type="password",
        value=st.session_state.get("rerank_api_key", ""),
        help="例如 Jina/Cohere 等托管重排服务的 Key。",
    )
    st.session_state["rerank_base_url"] = st.text_input(
        "Rerank Base URL",
        value=st.session_state.get(
            "rerank_base_url",
            "https://api.jina.ai/v1",
        )
        if st.session_state.get("rerank_provider", "jina") == "jina"
        else st.session_state.get("rerank_base_url", "https://api.cohere.com"),
        help="Jina 会拼接 /rerank，Cohere 会拼接 /v2/rerank。",
    )
    st.session_state["rerank_model"] = st.text_input(
        "Rerank Model",
        value=st.session_state.get(
            "rerank_model",
            "jina-reranker-v2-base-multilingual",
        )
        if st.session_state.get("rerank_provider", "jina") == "jina"
        else st.session_state.get("rerank_model", "rerank-v3.5"),
    )
    
    current_file_path = None
    df_info_str = "" 

    if uploaded_file:
        current_file_path = save_uploaded_file(uploaded_file)
        st.success(f"已加载: {uploaded_file.name}")
        try:
            if current_file_path.endswith(".csv"):
                df = pd.read_csv(current_file_path)
            else:
                df = pd.read_excel(current_file_path)
            df_info_str = get_dataframe_info(df)
            with st.expander("👀 数据透视"):
                st.text(df_info_str)
        except Exception as e:
            st.error(f"文件读取失败: {e}")
    if uploaded_pdf:
        pdf_path = save_uploaded_file(uploaded_pdf)
        st.success(f"知识库已加载: {uploaded_pdf.name}")
        
        if not embedding_api_key:
            st.error("❌ 请先在上方填入 Embedding API Key！")
            st.stop()

        try:
            with st.spinner("正在构建知识库索引 (向量 + BM25)..."):
                vector_db, bm25_retriever = create_vector_db(
                    pdf_path, embedding_api_key, embedding_base_url
                )

                st.session_state["vector_db"] = vector_db
                st.session_state["bm25_retriever"] = bm25_retriever
                st.success("✅ 知识库索引构建完成（支持混合检索）！")
        except Exception as e:
            st.error(f"知识库构建失败: {e}")
    # 将数据存入 Session 供 Tool 读取
    if current_file_path:
        st.session_state["current_file_path"] = current_file_path
    
    if st.button("🗑️ 清空当前会话历史"):
        # 这里只是简单清空 UI，数据库里的还在，除非手动删 DB 记录
        # 实际项目中可以调用 checkpointer 的删除方法，这里简单处理：换个 ID 就行
        st.session_state.messages = []
        st.rerun()

# 3. Agent 定义 (带 Checkpointer)
@st.cache_resource
def get_agent(_api_key):
    
    # --- 1. 定义增强版工具 (自动注入环境) ---
    @tool
    def python_interpreter(code: str):
        """
        [E2B 云端沙盒版] Python 代码执行器。
        代码将在安全的云端隔离环境中运行，支持 pandas, matplotlib 等库。
        """
        
        # 1. 检查 Key 是否存在
        if not e2b_api_key:
            return "❌ 错误: 未配置 E2B API Key，无法启动云端沙盒。"
            
        try:
            # 2. 启动云端沙盒 (使用 Context Manager 自动关闭)
            # 这一步需要几秒钟启动容器
            print("⏳ 正在启动 E2B 沙盒...")
            with Sandbox(api_key=e2b_api_key) as sbx:
                
                # --- A. 文件同步 (Local -> Cloud) ---
                # 我们需要把 Streamlit 里的数据文件上传到沙盒的 /home/user 目录下
                if "current_file_path" in st.session_state:
                    local_path = st.session_state["current_file_path"]
                    if os.path.exists(local_path):
                        with open(local_path, "rb") as f:
                            # 上传文件，并保持文件名一致
                            sbx.files.write(os.path.basename(local_path), f)
                            print(f"✅ 文件已上传至沙盒: {local_path}")
                
                # --- B. 代码预处理 (环境注入) ---
                clean_code = code.strip().replace("```python", "").replace("```", "")
                
                header = []
                header.append("import pandas as pd")
                header.append("import matplotlib.pyplot as plt")
                
                # 针对沙盒环境，我们需要设置非交互后端，且不用设置中文字体
                # (E2B 环境通常是 Linux，装中文字体比较麻烦，建议先用英文，或者上传字体文件)
                # 为了演示简单，我们先不强制设置字体，或者使用默认兼容设置
                header.append("plt.switch_backend('Agg')") 
                
                # 自动加载数据逻辑
                if "current_file_path" in st.session_state:
                    fname = os.path.basename(st.session_state["current_file_path"])
                    if fname.endswith(".csv"):
                        header.append(f"df = pd.read_csv('{fname}')")
                    elif fname.endswith((".xls", ".xlsx")):
                        header.append(f"df = pd.read_excel('{fname}')")

                full_code = "\n".join(header) + "\n" + clean_code
                
                # --- C. 执行代码 ---
                print(f"--- Cloud Executing ---\n{full_code}\n-----------------------")
                execution = sbx.run_code(full_code)
                
                # --- D. 处理结果 (Logs & Errors) ---
                output_log = ""
                if execution.error:
                    return f"❌ 代码执行报错:\n{execution.error.name}: {execution.error.value}\n{execution.error.traceback}"
                
                if execution.logs.stdout:
                    output_log += f"📄 输出:\n{str(execution.logs.stdout)}\n"
                
                # --- E. 处理图片 (Base64 -> Local PNG) ---
                # E2B 会把生成的图放在 execution.results 里
                for result in execution.results:
                    # 检查是否有 png 格式的输出
                    if hasattr(result, 'png') and result.png:
                        # 解码 Base64
                        img_data = base64.b64decode(result.png)
                        # 保存到 Streamlit 本地目录，以便前端显示
                        with open("result.png", "wb") as f:
                            f.write(img_data)
                        output_log += "\n🖼️ 图表已生成并保存为 result.png"
                
                if not output_log:
                    output_log = "代码执行成功，无文本输出。"
                    
                return output_log

        except Exception as e:
            return f"沙盒连接或执行失败: {e}"

    @tool
    def lookup_policy(query: str):
        """
        只有当用户询问具体的业务文档、政策、报告内容时才使用此工具。
        输入应该是具体的查询问题。
        """
        if "vector_db" not in st.session_state:
            return "用户还未上传文档，无法检索。"

        db = st.session_state["vector_db"]
        candidates_k = int(st.session_state.get("rag_candidates_k", 50))
        top_n = int(st.session_state.get("rag_top_n", 6))
        use_hybrid = bool(st.session_state.get("rag_hybrid", True))
        rrf_k = int(st.session_state.get("rag_rrf_k", 60))

        retriever = db.as_retriever(search_kwargs={"k": candidates_k})
        vec_docs = retriever.invoke(query) or []

        bm25 = st.session_state.get("bm25_retriever")
        if use_hybrid and bm25 is not None:
            bm25.k = candidates_k
            bm25_docs = bm25.invoke(query) or []
        else:
            bm25_docs = []

        if use_hybrid and vec_docs and bm25_docs:
            candidates = rrf_merge_doc_lists(vec_docs, bm25_docs, rrf_k=rrf_k)
        elif vec_docs:
            candidates = vec_docs
        elif bm25_docs:
            candidates = bm25_docs
        else:
            return "知识库中未找到相关信息。"

        # 托管 API 对单次 documents 条数常有限制，对 RRF 长列表截断后再精排
        merged = candidates
        rerank_cap = min(len(merged), max(2 * candidates_k, 20), 100)
        to_rerank = merged[:rerank_cap]

        # 托管 Rerank API（可选）。失败则保持 RRF/向量合并顺序。
        rerank_scores: list[float] | None = None
        final_ordered = merged
        rerank_api_key = str(st.session_state.get("rerank_api_key", "")).strip()
        rerank_base_url = str(st.session_state.get("rerank_base_url", "")).strip()
        rerank_model = str(st.session_state.get("rerank_model", "")).strip()
        rerank_provider = str(st.session_state.get("rerank_provider", "jina")).strip()
        if rerank_api_key and rerank_base_url and rerank_model and to_rerank:
            try:
                doc_texts = [d.page_content for d in to_rerank]
                scores = rerank_with_hosted_api(
                    query=query,
                    docs=doc_texts,
                    api_key=rerank_api_key,
                    base_url=rerank_base_url,
                    model=rerank_model,
                    provider=rerank_provider,
                )
                ranked = sorted(
                    zip(to_rerank, scores),
                    key=lambda x: x[1],
                    reverse=True,
                )
                rerank_ordered = [d for d, _ in ranked]
                rerank_scores = [float(s) for _, s in ranked]
                # 精排前缀 + 未送精排的 RRF 尾部
                final_ordered = rerank_ordered + merged[rerank_cap:]
            except Exception as e:
                st.warning(f"Rerank API 调用失败，已回退 RRF/向量排序：{e}")
                rerank_scores = None
                final_ordered = merged

        picked = final_ordered[:top_n]

        blocks = []
        for idx, doc in enumerate(picked, start=1):
            meta = getattr(doc, "metadata", {}) or {}
            source = os.path.basename(str(meta.get("source", ""))) or "unknown"
            page = meta.get("page", None)
            page_str = f"p.{int(page) + 1}" if isinstance(page, int) else "p.?"
            score_str = (
                f" | rerank={rerank_scores[idx - 1]:.4f}"
                if rerank_scores is not None and (idx - 1) < len(rerank_scores)
                else ""
            )
            blocks.append(
                f"[doc:{source} | {page_str} | chunk:{idx:02d}{score_str}]\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(blocks)
    tools = [python_interpreter, lookup_policy]

    llm = ChatOpenAI(
        api_key=_api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=0
    )
    llm_with_tools = llm.bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[List[BaseMessage], add_messages]

    def agent_node(state: AgentState):
        messages = state["messages"]
        
        # 动态获取 Prompt 数据
        file_path = st.session_state.get("current_file_path", "data.csv")
        data_context = df_info_str if df_info_str else "暂无数据"

        sys_msg = SystemMessage(content=f"""你是一个高级数据分析智能体，拥有两大核心能力：
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
        """)
        
        # 消息队列处理
        new_messages = list(messages)
        if isinstance(new_messages[0], SystemMessage):
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

    # 🔥【阶段3核心】初始化 SQLite 记忆数据库
    # check_same_thread=False 允许 Streamlit 多线程访问
    conn = sqlite3.connect("memory.sqlite", check_same_thread=False)
    memory = SqliteSaver(conn)
    
    # 编译时传入 checkpointer
    return workflow.compile(checkpointer=memory)

# 初始化 Agent
if api_key:
    app = get_agent(api_key)


# 从 LangGraph 数据库加载历史消息
# 我们不再依赖 st.session_state.messages 这种临时变量
# 而是直接去数据库里查这个 thread_id 有没有历史记录
current_config = {"configurable": {"thread_id": thread_id}}

if "messages" not in st.session_state:
    st.session_state.messages = []

# 尝试从 checkpointer 获取当前状态
try:
    snapshot = app.get_state(current_config)
    if snapshot.values and "messages" in snapshot.values:
        # 如果数据库里有记录，就显示数据库里的
        # 过滤掉 SystemMessage，只显示 User 和 AI
        st.session_state.messages = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in snapshot.values["messages"]
            if not isinstance(m, SystemMessage) and not isinstance(m, BaseMessage) # ToolMessage 这里简化处理不显示
            and m.content # 过滤空消息
        ]
except Exception as e:
    # 第一次可能是空的，忽略
    pass

# 显示历史
for msg in st.session_state.messages:
    # 简单过滤 ToolMessage (它的 role 可能是 tool)
    if msg["role"] in ["user", "assistant"]:
        st.chat_message(msg["role"]).write(msg["content"])

# 处理输入
if prompt := st.chat_input("输入你的分析需求..."):
    if not current_file_path and "vector_db" not in st.session_state:
        st.warning("⚠️ 你既没有上传数据，也没有上传知识库，我可能无法回答专业问题。")

    st.chat_message("user").write(prompt)

    st.chat_message("user").write(prompt)
    # 这里的 append 只是为了 UI 即使显示，真正的存储在 LangGraph 里
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        status_box = st.status("🧠 DeepSeek 正在回忆并思考...", expanded=True)
        
        try:
            # 直接把新消息丢给 app，带上 thread_id
            # LangGraph 会自动去数据库找之前的历史，拼在一起发给 DeepSeek
            inputs = {"messages": [HumanMessage(content=prompt)]}
            
            response_content = ""
            
            # 记得把 recursion_limit 调大
            events = app.stream(inputs, config=current_config)
            
            for event in events:
                if "agent" in event:
                    msg = event["agent"]["messages"][-1]
                    status_box.write(f"💬 思考: {msg.content}")
                    response_content = msg.content
                if "tools" in event:
                    tool_msg = event["tools"]["messages"][-1]
                    status_box.code(f"🛠️ 执行结果: {tool_msg.content[:300]}")

            status_box.update(label="✅ 完成", state="complete", expanded=False)
            
            if response_content:
                st.write(response_content)
                # 刷新 UI 状态
                st.session_state.messages.append({"role": "assistant", "content": response_content})
            
            if os.path.exists("result.png"):
                st.image("result.png")

        except Exception as e:
            st.error(f"发生错误: {e}")