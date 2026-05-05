from src.loader import load_pubmedqa
from src.indexer import build_vectorstore
from src.retriever import get_retriever
from src.pipeline import generate_answer

# 1. загрузка
docs, qa_pairs = load_pubmedqa("data/pubmedqa.json", limit=1000)

# 2. индекс
vectorstore = build_vectorstore(docs)

# 3. retriever
retriever = get_retriever(vectorstore, k=3)

# 4. тест
sample = qa_pairs[0]

query = sample["question"]

retrieved_docs = retriever.invoke(query)

answer = generate_answer(query, retrieved_docs)

print("QUESTION:", query)
print("ANSWER:", answer)