from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="qwen2.5:7b")

def generate_answer(query, docs):
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Ты научный ассистент.

Отвечай ТОЛЬКО на основе контекста.
Если ответа нет — напиши: "Нет информации в контексте".

Вопрос:
{query}

Контекст:
{context}

Ответ:
"""

    return llm.invoke(prompt)