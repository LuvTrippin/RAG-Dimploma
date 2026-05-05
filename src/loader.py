import json
from langchain_core.documents import Document


def load_pubmedqa(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    qa_pairs = []

    for i, (qid, item) in enumerate(data.items()):
        if limit is not None and i >= limit:
            break

        question = item["QUESTION"]
        contexts = item["CONTEXTS"]
        answer = item["final_decision"]
        long_answer = item.get("LONG_ANSWER", "")

        qa_pairs.append({
            "id": qid,
            "question": question,
            "answer": answer,
            "long_answer": long_answer,
            "relevant_contexts": len(contexts),
        })

        for context_index, ctx in enumerate(contexts):
            documents.append(
                Document(
                    page_content=ctx,
                    metadata={
                        "question_id": qid,
                        "context_index": context_index,
                        "source": "pubmedqa"
                    }
                )
            )

    return documents, qa_pairs
