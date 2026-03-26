from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional
import pickle

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


def _load_env() -> None:
    if load_dotenv is None:
        return
    # 默认加载项目根目录的 .env（你可以直接在这里填 key）
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=str(env_path), override=False)


_load_env()


def rerank_with_hosted_api(
    query: str,
    docs: list[str],
    api_key: str,
    base_url: str,
    model: str,
    provider: str,
    timeout_s: int = 30,
) -> list[float]:
    """
    调用托管 Rerank API 并返回与 docs 等长的分数。

    - provider == "jina": {base_url}/rerank
    - provider == "cohere": {base_url}/v2/rerank
    """
    if not docs:
        return []
    provider = provider.lower().strip()

    headers: dict[str, str] = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "query": query,
        "documents": docs,
        "top_n": len(docs),
    }

    if provider == "cohere":
        url = base_url.rstrip("/") + "/v2/rerank"
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        url = base_url.rstrip("/") + "/rerank"
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()

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


def _doc_dedupe_key(doc: Any) -> str:
    content = getattr(doc, "page_content", "") or ""
    return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()


def rrf_merge_doc_lists(*ranked_lists: list[Any], rrf_k: int = 60) -> list[Any]:
    """
    Reciprocal Rank Fusion: 多路有序结果合并为单一排序列表。
    ranked_lists 的元素应为 (ranked) Document 列表。
    """
    scores: dict[str, float] = {}
    doc_by_key: dict[str, Any] = {}

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


def _normalize_for_match(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def evidence_hit_ratio(
    predicted_contexts: list[Any],
    gold_entry: dict[str, Any],
) -> tuple[float, int, int]:
    """
    粗粒度证据命中：gold 的每个 source_text 片段，是否能在预测上下文中找到子串。

    返回 (ratio, hits, needed)
    """
    gold_source_text = gold_entry.get("source_text")
    if gold_source_text is None:
        return 0.0, 0, 0

    if isinstance(gold_source_text, str):
        gold_snips: list[str] = [gold_source_text]
    elif isinstance(gold_source_text, list):
        gold_snips = [x for x in gold_source_text if isinstance(x, str)]
    else:
        gold_snips = []

    needed = int(gold_entry.get("num_sources_used", len(gold_snips) or 0))
    needed = max(needed, 1) if gold_snips else 0
    if not gold_snips:
        return 0.0, 0, 0

    predicted_text = _normalize_for_match("\n".join(getattr(d, "page_content", "") or "" for d in predicted_contexts))
    hits = 0
    for snip in gold_snips:
        sn = _normalize_for_match(snip)
        if not sn:
            continue
        if len(sn) <= 120:
            if sn in predicted_text:
                hits += 1
        else:
            # 对过长片段，用更小的前缀进行兜底
            head = sn[:200]
            if head in predicted_text:
                hits += 1

    return (hits / needed if needed else 0.0), hits, needed


def normalize_text_for_scoring(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    # 保留中英文/数字/基础符号，减少“标点导致的差异”
    s = re.sub(r"[^0-9a-z\u4e00-\u9fff\.\-\+ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_f1(pred: str, gold: str) -> float:
    pred_n = normalize_text_for_scoring(pred)
    gold_n = normalize_text_for_scoring(gold)

    pred_tokens = pred_n.split()
    gold_tokens = gold_n.split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    gold_set = {}
    for t in gold_tokens:
        gold_set[t] = gold_set.get(t, 0) + 1

    common = 0
    for t in pred_tokens:
        if t in gold_set and gold_set[t] > 0:
            common += 1
            gold_set[t] -= 1

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def first_number(s: str) -> Optional[float]:
    m = re.findall(r"[-+]?\d+(?:\.\d+)?", s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def answer_score(pred: str, gold: str) -> float:
    """
    简单答案打分：
    - 若可解析首个数字，则用数值相近度（相对误差）
    - 否则用 token_f1
    """
    pred_n = pred.strip()
    gold_n = gold.strip()
    pn = first_number(pred_n)
    gn = first_number(gold_n)
    if pn is not None and gn is not None:
        if gn == 0:
            return 1.0 if pn == 0 else 0.0
        rel = abs(pn - gn) / abs(gn)
        return 1.0 if rel <= 0.01 else max(0.0, 1.0 - rel)
    return token_f1(pred_n, gold_n)


def _safe_extract_json(text: str) -> dict[str, Any]:
    """
    允许 LLM 返回文本包裹 JSON 的情况，尽量提取第一段 JSON 对象。
    """
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return {}


def truncate_contexts(contexts: list[Any], *, max_chars_per_ctx: int = 800) -> str:
    """
    将 contexts 压缩成裁判可读且不爆 token 的文本。
    """
    parts: list[str] = []
    for i, d in enumerate(contexts, start=1):
        meta = getattr(d, "metadata", {}) or {}
        src = os.path.basename(str(meta.get("source", "") or "unknown"))
        page = meta.get("page", None)
        page_str = f"p.{int(page) + 1}" if isinstance(page, int) else "p.?"
        content = getattr(d, "page_content", "") or ""
        content = content[:max_chars_per_ctx]
        parts.append(f"[{i}] {src} {page_str}\n{content}")
    return "\n\n".join(parts)


def hybrid_llm_judge(
    judge_llm: ChatOpenAI,
    *,
    question: str,
    gold_answer: str,
    pred_answer: str,
    contexts: list[Any],
    hard_evidence_ratio: float,
) -> dict[str, Any]:
    """
    混合裁判：
    - hard_evidence_ratio：你现有的“搜得到/引用得到”硬匹配证据命中率
    - semantic_correctness：LLM judge 判断意思是否对（允许同义/改写）
    - citation_grounding：LLM judge 判断回答是否能从 contexts 支撑

    输出统一结构，便于后续做指标对比。
    """
    contexts_text = truncate_contexts(contexts)
    system = SystemMessage(
        content=(
            "你是严谨的 RAG 评测裁判。你将看到：问题、gold答案、模型预测答案、以及检索到的证据上下文。"
            "请基于证据上下文判断：预测答案的语义是否与 gold 一致（允许改写同义），以及预测答案是否被上下文充分支撑。"
            "输出严格 JSON，不要输出除 JSON 之外的任何文本。"
        )
    )
    human = HumanMessage(
        content=(
            f"【问题】\n{question}\n\n"
            f"【gold答案】\n{gold_answer}\n\n"
            f"【模型预测答案】\n{pred_answer}\n\n"
            f"【检索证据上下文（可能不完整）】\n{contexts_text}\n\n"
            f"【硬证据命中率 hard_evidence_ratio】\n{hard_evidence_ratio:.4f}\n\n"
            "请按以下字段输出 JSON：\n"
            "- evidence_grounded: 0-1（预测答案是否能被证据上下文支撑）\n"
            "- semantic_correctness: 0-1（预测答案与 gold 的语义是否一致；允许同义改写）\n"
            "- keyword_string_match: 0-1（预测答案中是否包含 gold 的关键实体/数字/专有名词；近似即可）\n"
            "- hybrid_score: 0-1（总体分，建议用 0.4*evidence_grounded + 0.6*semantic_correctness）\n"
            "- notes: string（一句话说明关键原因）"
        )
    )

    resp = judge_llm.invoke([system, human]).content or ""
    parsed = _safe_extract_json(resp)

    # 兜底：防止 judge 输出不符合预期导致程序崩溃
    def _get_float(key: str, default: float = 0.0) -> float:
        v = parsed.get(key, default)
        try:
            return float(v)
        except Exception:
            return default

    return {
        "evidence_grounded": _get_float("evidence_grounded", 0.0),
        "semantic_correctness": _get_float("semantic_correctness", 0.0),
        "keyword_string_match": _get_float("keyword_string_match", 0.0),
        "hybrid_score": _get_float("hybrid_score", 0.0),
        "notes": str(parsed.get("notes", "")),
        "raw_judge_text": resp[:2000],
    }


@dataclass
class PredictedContext:
    text: str
    source: str
    page: Any


def doc_to_citation(doc: Any, idx: int) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    src = os.path.basename(str(meta.get("source", "") or "unknown"))
    page = meta.get("page", None)
    page_str = f"p.{int(page) + 1}" if isinstance(page, int) else "p.?"
    return f"[{idx:02d}] {src} {page_str}\n{getattr(doc, 'page_content', '')}"


@lru_cache(maxsize=8)
def build_index_cached(
    pdf_path: str,
    embedding_api_key: str,
    embedding_base_url: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
):
    project_root = Path(__file__).resolve().parent
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=embedding_api_key,
        openai_api_base=embedding_base_url,
    )
    persist_hash = hashlib.sha256(
        (
            pdf_path
            + embedding_api_key
            + embedding_base_url
            + embedding_model
            + str(chunk_size)
            + str(chunk_overlap)
        ).encode("utf-8", errors="ignore")
    ).hexdigest()[:16]
    persist_dir = str(project_root / ".chroma_eval" / persist_hash)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="pdfqa_eval",
    )

    # BM25 磁盘缓存：把 BM25Okapi + docs 列表 pickle
    bm25_dir = project_root / ".bm25_eval" / persist_hash
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
        bm25_retriever = BM25Retriever.from_documents(splits, k=50)
    return vectorstore, bm25_retriever


def find_pdf_path(
    pdfs_root: str,
    category: str,
    domain: str,
    file_name: str,
) -> Optional[str]:
    """
    在 pdfQA-Benchmark 目录下查找对应 PDF。
    尝试：
    - {pdfs_root}/{category}/01.2_Input_Files_PDF/{domain}/{file_name}
    - 如果 file_name 不带 .pdf，则补上
    """
    base = os.path.join(
        pdfs_root,
        category,
        "01.2_Input_Files_PDF",
        domain,
    )
    c1 = os.path.join(base, file_name)
    if os.path.isfile(c1):
        return c1
    if not file_name.lower().endswith(".pdf"):
        c2 = c1 + ".pdf"
        if os.path.isfile(c2):
            return c2
    # 兜底：只用文件名 tail 匹配（少量 PDF 时可用）
    tail = os.path.basename(file_name)
    for root, _, files in os.walk(base):
        for fn in files:
            if fn == tail or fn == tail + ".pdf":
                return os.path.join(root, fn)
    return None


def iter_annotation_examples(
    annotations_root: str,
    category: str,
    domain: str,
    max_examples: int,
) -> Iterable[dict[str, Any]]:
    """
    遍历 pdfQA-Annotations 下 JSON 条目。
    """
    cat_dir = os.path.join(annotations_root, category, domain)
    if not os.path.isdir(cat_dir):
        # 有些数据组织可能多一层 dataset name，这里尝试宽松兜底
        cat_dir = os.path.join(annotations_root, category)
        if not os.path.isdir(cat_dir):
            raise FileNotFoundError(f"annotations 目录不存在：{cat_dir}")

    count = 0
    for root, _, files in os.walk(cat_dir):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            fp = os.path.join(root, fn)
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                entries = data["data"]
            elif isinstance(data, list):
                entries = data
            else:
                # 不确定结构就跳过
                continue
            if not entries:
                continue
            for e in entries:
                if not isinstance(e, dict):
                    continue
                yield e
                count += 1
                if count >= max_examples:
                    return


def process_single_example(
    entry: dict[str, Any],
    args: argparse.Namespace,
    llm: ChatOpenAI | None,
    judge_llm: ChatOpenAI | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """
    处理单条测试样本：检索（Hybrid + rerank 可选）-> 证据命中 ->（可选）生成 ->（可选）Hybrid Judge

    返回：
      - per_example_dict 或 None（表示跳过）
      - metrics_dict 或 None
    """
    idx = entry.get("_idx")
    question = entry.get("question", "")
    gold_answer = entry.get("answer", "")
    file_name = entry.get("file_name", entry.get("file", entry.get("document", "")))

    if not file_name:
        return None, None

    pdf_path = find_pdf_path(
        pdfs_root=args.pdfs_root,
        category=args.category,
        domain=args.domain,
        file_name=str(file_name),
    )
    if not pdf_path:
        return None, None

    vectorstore, bm25_retriever = build_index_cached(
        pdf_path=pdf_path,
        embedding_api_key=args.embedding_api_key,
        embedding_base_url=args.embedding_base_url,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    vec_docs = vectorstore.similarity_search(question, k=args.candidates_k) or []
    bm25_retriever.k = args.candidates_k
    bm25_docs = bm25_retriever.get_relevant_documents(question) or []

    if vec_docs or bm25_docs:
        merged = rrf_merge_doc_lists(vec_docs, bm25_docs, rrf_k=args.rrf_k)
    else:
        merged = []

    if not merged:
        return None, None

    rerank_cap = min(len(merged), max(2 * args.candidates_k, 20), 100)
    to_rerank = merged[:rerank_cap]

    final_order = merged
    rerank_scores: list[float] | None = None
    if args.enable_rerank and args.rerank_api_key and to_rerank:
        try:
            doc_texts = [d.page_content for d in to_rerank]
            rerank_scores = rerank_with_hosted_api(
                query=question,
                docs=doc_texts,
                api_key=args.rerank_api_key,
                base_url=args.rerank_base_url,
                model=args.rerank_model,
                provider=args.rerank_provider,
            )
            ranked = sorted(zip(to_rerank, rerank_scores), key=lambda x: x[1], reverse=True)
            rerank_order = [d for d, _ in ranked]
            final_order = rerank_order + merged[rerank_cap:]
        except Exception:
            final_order = merged

    contexts = final_order[: args.top_n]
    ratio, hits, needed = evidence_hit_ratio(contexts, entry)

    pred_answer = ""
    answer_score_val: float | None = None
    if llm is not None:
        citation_blocks = [doc_to_citation(d, i + 1) for i, d in enumerate(contexts)]
        context_text = "\n\n".join(citation_blocks)
        sys = SystemMessage(
            content=(
                "你是一个严谨的 PDF 问答助手。"
                "只允许基于【提供的证据上下文】回答。"
                "如果证据不足，回答“知识库中未找到相关信息”。"
                "回答中要给出至少一个证据引用编号（如 [01]）。"
            )
        )
        human = HumanMessage(
            content=f"【问题】{question}\n\n【证据上下文】\n{context_text}\n\n请回答："
        )
        pred_answer = llm.invoke([sys, human]).content
        answer_score_val = answer_score(pred_answer, str(gold_answer))

    # Hybrid Judge：语义正确性（意思对不对）+ 忠实引用（引用是否可靠）
    hybrid_judge: dict[str, Any] | None = None
    hybrid_judge_score: float | None = None
    semantic_correctness_score: float | None = None
    if judge_llm is not None and llm is not None:
        hybrid_judge = hybrid_llm_judge(
            judge_llm,
            question=question,
            gold_answer=str(gold_answer),
            pred_answer=pred_answer,
            contexts=contexts,
            hard_evidence_ratio=ratio,
        )
        hybrid_judge_score = float(hybrid_judge.get("hybrid_score", 0.0))
        semantic_correctness_score = float(hybrid_judge.get("semantic_correctness", 0.0))

    per_example = {
        "id": idx,
        "question": question,
        "gold_answer": gold_answer,
        "pred_answer": pred_answer,
        "file_name": file_name,
        "pdf_path": pdf_path,
        "evidence_hit_ratio": ratio,
        "evidence_hits": hits,
        "evidence_needed": needed,
        "contexts": [
            {
                "source": os.path.basename(str(getattr(d, "metadata", {}) or {}).get("source", "unknown")),
                "page": (getattr(d, "metadata", {}) or {}).get("page", None),
                "text_len": len(getattr(d, "page_content", "") or ""),
            }
            for d in contexts
        ],
        "rerank_scores": rerank_scores,
        "hybrid_judge": hybrid_judge,
    }

    metrics = {
        "evidence_ratio": ratio,
        "evidence_hits": hits,
        "evidence_needed": needed,
        "answer_score": answer_score_val,
        "hybrid_judge_score": hybrid_judge_score,
        "semantic_correctness_score": semantic_correctness_score,
    }
    return per_example, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--annotations_root",
        default=os.getenv("ANNOTATIONS_ROOT", ""),
        help="pdfqa/pdfQA-Annotations 本地根目录（也可从 .env 的 ANNOTATIONS_ROOT 读取）",
    )
    ap.add_argument(
        "--pdfs_root",
        default=os.getenv("PDFS_ROOT", ""),
        help="pdfqa/pdfQA-Benchmark 本地根目录（也可从 .env 的 PDFS_ROOT 读取）",
    )
    ap.add_argument(
        "--category",
        default=os.getenv("PDFQA_CATEGORY", "real-pdfQA"),
        choices=["real-pdfQA", "syn-pdfQA"],
    )
    ap.add_argument(
        "--domain",
        default=os.getenv("PDFQA_DOMAIN", "ClimateFinanceBench"),
        help="例如 ClimateFinanceBench / FinQA / Tat-QA ...",
    )

    ap.add_argument("--max_examples", type=int, default=int(os.getenv("MAX_EXAMPLES", "30")))
    ap.add_argument("--candidates_k", type=int, default=int(os.getenv("CANDIDATES_K", "80")), help="每路召回 k（向量/BM25 各一遍）")
    ap.add_argument("--top_n", type=int, default=int(os.getenv("TOP_N", "6")), help="最终上下文片段数")
    ap.add_argument("--rrf_k", type=int, default=int(os.getenv("RRF_K", "60")))

    ap.add_argument("--embedding_api_key", default=os.getenv("EMBEDDING_API_KEY", ""))
    ap.add_argument(
        "--embedding_base_url",
        default=os.getenv("EMBEDDING_BASE_URL", "https://poloai.top/v1"),
    )
    ap.add_argument("--embedding_model", default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    ap.add_argument("--chunk_size", type=int, default=int(os.getenv("CHUNK_SIZE", "1000")))
    ap.add_argument("--chunk_overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "200")))

    ap.add_argument("--rerank_api_key", default=os.getenv("RERANK_API_KEY", ""))
    ap.add_argument(
        "--rerank_base_url",
        default=os.getenv("RERANK_BASE_URL", "https://api.jina.ai/v1"),
    )
    ap.add_argument(
        "--rerank_model",
        default=os.getenv("RERANK_MODEL", "jina-reranker-v2-base-multilingual"),
    )
    ap.add_argument(
        "--rerank_provider",
        default=os.getenv("RERANK_PROVIDER", "jina"),
        choices=["jina", "cohere"],
    )
    ap.add_argument("--enable_rerank", action="store_true")

    ap.add_argument("--generate_answers", action="store_true", help="是否生成最终答案（会额外调用 LLM）")
    ap.add_argument(
        "--llm_api_key",
        default=os.getenv("LLM_API_KEY", ""),
        help="生成回答用 API Key；不填则禁用生成",
    )
    ap.add_argument(
        "--llm_base_url",
        default=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
    )
    ap.add_argument("--llm_model", default=os.getenv("LLM_MODEL", "deepseek-chat"))

    ap.add_argument("--enable_hybrid_judge", action="store_true", help="启用 Hybrid Judge（LLM 语义裁判 + 硬匹配）")
    ap.add_argument(
        "--judge_api_key",
        default=os.getenv("JUDGE_API_KEY", ""),
        help="裁判模型 API Key；不填则复用 --llm_api_key",
    )
    ap.add_argument(
        "--judge_base_url",
        default=os.getenv("JUDGE_BASE_URL", "https://api.deepseek.com"),
    )
    ap.add_argument("--judge_model", default=os.getenv("JUDGE_MODEL", "deepseek-chat"))
    ap.add_argument(
        "--judge_temperature",
        type=float,
        default=float(os.getenv("JUDGE_TEMPERATURE", "0.0")),
    )
    args = ap.parse_args()

    if not args.annotations_root:
        raise ValueError("annotations_root 为空：请在命令行传入 --annotations_root，或在 .env 设置 ANNOTATIONS_ROOT")
    if not args.pdfs_root:
        raise ValueError("pdfs_root 为空：请在命令行传入 --pdfs_root，或在 .env 设置 PDFS_ROOT")

    if not args.embedding_api_key:
        raise ValueError(
            "embedding_api_key 为空：请在 .env 中配置 EMBEDDING_API_KEY，或在命令行传入 --embedding_api_key"
        )

    if args.generate_answers and not args.llm_api_key:
        raise ValueError("--generate_answers 需要 --llm_api_key")

    llm = None
    if args.generate_answers:
        llm = ChatOpenAI(
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
            model=args.llm_model,
            temperature=0,
        )

    judge_llm: ChatOpenAI | None = None
    if args.enable_hybrid_judge:
        if not args.llm_api_key and not args.judge_api_key:
            raise ValueError("--enable_hybrid_judge 需要 --generate_answers 或至少提供 --llm_api_key/--judge_api_key")
        judge_api_key = args.judge_api_key or args.llm_api_key
        judge_llm = ChatOpenAI(
            api_key=judge_api_key,
            base_url=args.judge_base_url,
            model=args.judge_model,
            temperature=args.judge_temperature,
        )

    per_example: list[dict[str, Any]] = []
    evidence_ratios: list[float] = []
    evidence_hits: list[int] = []
    evidence_needed: list[int] = []
    answer_scores: list[float] = []
    hybrid_judge_scores: list[float] = []
    semantic_correctness_scores: list[float] = []

    for idx, entry in enumerate(
        iter_annotation_examples(
            annotations_root=args.annotations_root,
            category=args.category,
            domain=args.domain,
            max_examples=args.max_examples,
        ),
        start=1,
    ):
        entry_with_idx = dict(entry)
        entry_with_idx["_idx"] = idx
        per_example_item, metrics = process_single_example(
            entry_with_idx,
            args=args,
            llm=llm,
            judge_llm=judge_llm,
        )

        if per_example_item is None or metrics is None:
            continue

        per_example.append(per_example_item)
        evidence_ratios.append(float(metrics["evidence_ratio"]))
        evidence_hits.append(int(metrics["evidence_hits"]))
        evidence_needed.append(int(metrics["evidence_needed"]))

        if metrics["answer_score"] is not None:
            answer_scores.append(float(metrics["answer_score"]))
        if metrics["hybrid_judge_score"] is not None:
            hybrid_judge_scores.append(float(metrics["hybrid_judge_score"]))
        if metrics["semantic_correctness_score"] is not None:
            semantic_correctness_scores.append(float(metrics["semantic_correctness_score"]))

    total = {
        "n_examples": len(per_example),
        "evidence_recall_avg": sum(evidence_ratios) / len(evidence_ratios) if evidence_ratios else 0.0,
        "evidence_hits_sum": sum(evidence_hits),
        "evidence_needed_sum": sum(evidence_needed),
        "answer_score_avg": sum(answer_scores) / len(answer_scores) if answer_scores else None,
        "hybrid_judge_score_avg": sum(hybrid_judge_scores) / len(hybrid_judge_scores)
        if hybrid_judge_scores
        else None,
        "semantic_correctness_avg": sum(semantic_correctness_scores) / len(semantic_correctness_scores)
        if semantic_correctness_scores
        else None,
    }

    out = {"summary": total, "per_example": per_example}
    with open("eval_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("✅ Evaluation finished.")
    print(json.dumps(total, ensure_ascii=False, indent=2))
    print("Report saved to: eval_report.json")


if __name__ == "__main__":
    main()

