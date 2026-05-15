from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MIN_RELEVANCE_SCORE,
    TOP_K_RESULTS,
)


RETRIEVER_TOOL = {
    "name": "retrieve_context",
    "description": "Search the knowledge base for relevant information to answer the user's question. Use this whenever the question may depend on ingested documents.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for retrieving relevant context",
            }
        },
        "required": ["query"],
    },
}

NO_CONTEXT_RESPONSE = "No relevant context found in the knowledge base."

_embedding_model = None
STOPWORDS = {
    "about",
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "do",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "many",
    "of",
    "on",
    "or",
    "please",
    "related",
    "said",
    "say",
    "tell",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "who",
    "why",
}


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def normalize_query(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def get_source_term_map() -> dict[str, str]:
    try:
        collection = get_collection()
        if collection.count() == 0:
            return {}

        results = collection.get(include=["metadatas"])
    except Exception:
        return {}

    term_map = {}
    for metadata in results.get("metadatas", []):
        filename = metadata.get("filename", "").strip().lower()
        source_stem = metadata.get("source_stem", "").strip().lower()
        if filename:
            term_map[filename] = filename
        if source_stem:
            term_map[source_stem] = filename

    return term_map


def get_known_source_terms() -> set[str]:
    return set(get_source_term_map())


def detect_source_filter(query: str) -> str | None:
    normalized = normalize_query(query)
    source_map = get_source_term_map()
    for term, filename in source_map.items():
        if term and term in normalized:
            return filename
    return None


def rewrite_query(query: str) -> str:
    source_filter = detect_source_filter(query)
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    keywords = [token for token in tokens if token not in STOPWORDS and len(token) > 2]
    if not keywords:
        keywords = tokens[:]

    if source_filter:
        source_stem = Path(source_filter).stem.lower()
        if source_stem not in keywords:
            keywords.insert(0, source_stem)

    rewritten = " ".join(dict.fromkeys(keywords))
    return rewritten.strip() or query.strip()


def keyword_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 2
    }


def format_context(documents: list[str], metadatas: list[dict], scores: list[float]) -> str:

    lines = ["--- Retrieved Context ---"]
    for document, metadata, score in zip(documents, metadatas, scores):
        filename = metadata.get("filename", "unknown")
        page_number = metadata.get("page_number", "unknown")
        chunk_index = metadata.get("chunk_index", "unknown")
        lines.append(
            f"[Source: {filename} | Page {page_number} | Chunk {chunk_index} | Score {score:.2f}]"
        )
        lines.append(document)
        lines.append("")

    lines.append("-------------------------")
    return "\n".join(lines)


def retrieve(query: str, verbose: bool = False) -> str:
    if not query.strip():
        if verbose:
            print("[retriever] Empty query. No vector search performed.")
        return NO_CONTEXT_RESPONSE

    try:
        collection = get_collection()
        if collection.count() == 0:
            if verbose:
                print("[retriever] Collection is empty. No vector search results.")
            return NO_CONTEXT_RESPONSE

        source_filter = detect_source_filter(query)
        rewritten_query = rewrite_query(query)
        if verbose:
            print(f"[retriever] Original query: {query}")
            print(f"[retriever] Rewritten query: {rewritten_query}")
            if source_filter:
                print(f"[retriever] Applying source filter: {source_filter}")

        embedding = get_embedding_model().encode(rewritten_query).tolist()
        query_args = {
            "query_embeddings": [embedding],
            "n_results": min(TOP_K_RESULTS, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if source_filter:
            query_args["where"] = {"filename": source_filter}

        results = collection.query(**query_args)
    except Exception as exc:
        if verbose:
            print(f"[retriever] Retrieval error: {exc}")
        return NO_CONTEXT_RESPONSE

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    if not documents:
        if verbose:
            print("[retriever] Vector DB returned 0 matching chunks.")
        return NO_CONTEXT_RESPONSE

    scores = [1.0 / (1.0 + float(distance)) for distance in distances]
    top_score = max(scores) if scores else 0.0
    if top_score < MIN_RELEVANCE_SCORE:
        if verbose:
            print(
                f"[retriever] Retrieval confidence too low. Best score: {top_score:.2f}"
            )
        return NO_CONTEXT_RESPONSE

    query_keywords = keyword_tokens(rewritten_query)
    filtered_rows = []
    for document, metadata, score in zip(documents, metadatas, scores):
        overlap = query_keywords & keyword_tokens(document)
        if overlap:
            filtered_rows.append((document, metadata, score, overlap))

    if not filtered_rows:
        if verbose:
            print("[retriever] Retrieved chunks had no keyword overlap with the query.")
        return NO_CONTEXT_RESPONSE

    documents = [row[0] for row in filtered_rows]
    metadatas = [row[1] for row in filtered_rows]
    scores = [row[2] for row in filtered_rows]

    if verbose:
        print(
            f"[retriever] Vector DB returned {len(documents)} chunk(s). "
            f"Best score: {max(scores):.2f}"
        )
        for metadata, score, (_, _, _, overlap) in zip(metadatas, scores, filtered_rows):
            filename = metadata.get("filename", "unknown")
            page_number = metadata.get("page_number", "unknown")
            chunk_index = metadata.get("chunk_index", "unknown")
            print(
                f"[retriever] - {filename} | page {page_number} | chunk {chunk_index} "
                f"| score {score:.2f} | overlap {sorted(overlap)}"
            )

    return format_context(documents, metadatas, scores)
