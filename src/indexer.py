import hashlib
import json
import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


PERSIST_DIR = "./chroma_db_v1"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
MANIFEST_NAME = "manifest.json"


def _documents_hash(documents):
    digest = hashlib.sha256()

    for doc in documents:
        digest.update(doc.page_content.encode("utf-8"))
        digest.update(
            json.dumps(doc.metadata, sort_keys=True, ensure_ascii=False).encode("utf-8")
        )

    return digest.hexdigest()


def _manifest_path(persist_dir):
    return os.path.join(persist_dir, MANIFEST_NAME)


def _load_manifest(persist_dir):
    path = _manifest_path(persist_dir)
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(persist_dir, manifest):
    os.makedirs(persist_dir, exist_ok=True)
    with open(_manifest_path(persist_dir), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def build_or_load_vectorstore(
    documents,
    persist_dir=PERSIST_DIR,
    embedding_model=DEFAULT_EMBEDDING_MODEL,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    rebuild=False,
):
    embeddings = OllamaEmbeddings(model=embedding_model)
    manifest = {
        "documents_hash": _documents_hash(documents),
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "documents_count": len(documents),
    }

    existing_manifest = _load_manifest(persist_dir)
    has_index = os.path.exists(persist_dir) and bool(os.listdir(persist_dir))

    if has_index and not rebuild and existing_manifest == manifest:
        print("Loading existing vector store...")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )

    if has_index:
        print("Rebuilding vector store because index parameters or data changed...")
        shutil.rmtree(persist_dir)

    print("Creating new vector store...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    split_docs = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        split_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    _write_manifest(persist_dir, manifest)

    return vectorstore
