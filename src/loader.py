import json
from langchain_core.documents import Document

def load_pubmedqa(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    qa_pairs = []

    for i, (qid, item) in enumerate(data.items()):
        if limit and i >= limit:
            break

        question = item["QUESTION"]
        contexts = item["CONTEXTS"]
        answer = item["final_decision"]

        # сохраняем QA
        qa_pairs.append({
            "id": qid,
            "question": question,
            "answer": answer
        })

        # создаём документы
        for ctx in contexts:
            documents.append(
                Document(
                    page_content=ctx,
                    metadata={
                        "question_id": qid,
                        "source": "pubmedqa"
                    }
                )
            )

    return documents, qa_pairs