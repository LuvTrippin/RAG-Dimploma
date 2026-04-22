from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. Документы
docs = [
    Document(page_content="Фотогенез это процесс, при котором растения преобразуют свет в энергию."),
    Document(page_content="Альберт Эйнштейн родился 14 марта 1879 года в городе Ульм, Германия."),
    Document(page_content="Митохондрия это энергетическая станция клетки."),
]

# 2. Сплит
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

# 3. Embeddings (можно оставить qwen, но лучше отдельную модель)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 4. Векторная база
vectorstore = Chroma.from_documents(split_docs, embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 5. LLM
llm = OllamaLLM(model="qwen:7b")

# 6. Pipeline
def ask(query):
    docs = retriever.invoke(query)
    
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    Используй ТОЛЬКО предоставленный текст. 
    Если информации нет — не пытайся придумать ответ.

    Контекст:
    {context}

    Вопрос:
    {query}
    """

    print("Prompt:\n", prompt)
    
    return llm.invoke(prompt)

# тест
print(ask("Что такое митохондрия?"))