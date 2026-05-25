from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.indexer import DEFAULT_EMBEDDING_MODEL

from .file_parsers import extract_text


UPLOADS_DIR = Path("webapp_data/uploads")
INDEX_DIR = Path("webapp_data/chroma_web")
CATALOG_PATH = Path("webapp_data/catalog.json")
DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 120


def ensure_storage() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CATALOG_PATH.exists():
        CATALOG_PATH.write_text("[]", encoding="utf-8")


def _load_catalog() -> list[dict]:
    ensure_storage()
    with CATALOG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_catalog(items: list[dict]) -> None:
    CATALOG_PATH.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _make_store(embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> Chroma:
    embeddings = OllamaEmbeddings(model=embedding_model)
    return Chroma(
        persist_directory=str(INDEX_DIR),
        embedding_function=embeddings,
    )


def list_sources() -> list[dict]:
    return _load_catalog()


def _resolve_source_path(source_path: str) -> Path:
    resolved = Path(source_path).resolve()
    uploads_root = UPLOADS_DIR.resolve()
    try:
        resolved.relative_to(uploads_root)
    except ValueError as exc:
        raise RuntimeError("Source path is outside uploads storage.") from exc
    return resolved


def get_source_content(source_path: str, max_chars: int = 30000) -> dict:
    path = _resolve_source_path(source_path)
    if not path.exists() or not path.is_file():
        raise RuntimeError("Source file not found.")

    content = extract_text(path)
    if not content:
        content = "(Пустой файл)"
    return {
        "source_name": path.name,
        "source_path": str(path),
        "content": content[:max_chars],
        "truncated": len(content) > max_chars,
    }


def ingest_files(
    file_paths: Iterable[Path],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    ensure_storage()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    store = _make_store(embedding_model=embedding_model)
    catalog = _load_catalog()

    source_entries = []
    all_chunks: list[Document] = []

    for source_path in file_paths:
        text = extract_text(source_path)
        if not text:
            continue

        source_id = source_path.stem
        base_document = Document(
            page_content=text,
            metadata={
                "source_name": source_path.name,
                "source_path": str(source_path),
                "source_id": source_id,
                "ingested_via": "webapp",
            },
        )
        chunks = splitter.split_documents([base_document])
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
        all_chunks.extend(chunks)
        source_entries.append(
            {
                "source_name": source_path.name,
                "source_path": str(source_path),
                "source_id": source_id,
                "chunks": len(chunks),
            }
        )

    if all_chunks:
        store.add_documents(all_chunks)
        catalog.extend(source_entries)
        _save_catalog(catalog)

    return {
        "sources_added": len(source_entries),
        "chunks_added": len(all_chunks),
    }


def ask(
    query: str,
    llm,
    history: list[dict] | None = None,
    k: int = 4,
    max_context_chars: int = 5000,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict:
    store = _make_store(embedding_model=embedding_model)
    retriever = store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)

    from src.pipeline import generate_answer

    history = history or []
    if history:
        recent = history[-6:]
        history_block = "\n".join(
            f"{item.get('role', 'user')}: {item.get('text', '')}" for item in recent
        )
        prompt_query = (
            "История диалога:\n"
            f"{history_block}\n\n"
            "Новый вопрос пользователя:\n"
            f"{query}\n\n"
            "Учитывай только релевантную часть истории и не придумывай факты."
        )
    else:
        prompt_query = query

    answer = generate_answer(
        query=prompt_query,
        docs=docs,
        llm=llm,
        max_context_chars=max_context_chars,
    )

    sources = []
    for doc in docs:
        snippet = doc.page_content[:250].replace("\n", " ").strip()
        sources.append(
            {
                "source_name": doc.metadata.get("source_name", "unknown"),
                "source_path": doc.metadata.get("source_path"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "snippet": snippet,
            }
        )

    return {"answer": answer, "sources": sources}
