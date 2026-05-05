from langchain_ollama import OllamaEmbeddings


def is_relevant(doc, qid):
    return doc.metadata.get("question_id") == qid


def hit_at_k(retrieved_docs, qid):
    return int(any(is_relevant(doc, qid) for doc in retrieved_docs))


def recall_at_k(retrieved_docs, qid, relevant_contexts):
    if relevant_contexts <= 0:
        return 0

    retrieved_contexts = {
        doc.metadata.get("context_index")
        for doc in retrieved_docs
        if is_relevant(doc, qid) and doc.metadata.get("context_index") is not None
    }
    return len(retrieved_contexts) / relevant_contexts


def precision_at_k(retrieved_docs, qid):
    if not retrieved_docs:
        return 0
    relevant = sum(is_relevant(doc, qid) for doc in retrieved_docs)
    return relevant / len(retrieved_docs)


def classification_accuracy(predicted, expected):
    return int(predicted == expected)


def _cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0

    return dot / (norm1 * norm2)


def semantic_similarity(text1, text2, embeddings=None, embedding_model="nomic-embed-text"):
    if not text1 or not text2:
        return 0

    embeddings = embeddings or OllamaEmbeddings(model=embedding_model)
    vec1, vec2 = embeddings.embed_documents([text1, text2])
    return _cosine_similarity(vec1, vec2)
