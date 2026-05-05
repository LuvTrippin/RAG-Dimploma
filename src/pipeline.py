from langchain_ollama import OllamaLLM


VALID_DECISIONS = {"yes", "no", "maybe"}


def create_llm(model="qwen:7b"):
    return OllamaLLM(model=model)


def _build_context(docs, max_chars=4000):
    context = "\n\n".join(d.page_content for d in docs)
    return context[:max_chars]


def generate_answer(query, docs, llm=None, max_context_chars=4000):
    llm = llm or create_llm()
    context = _build_context(docs, max_context_chars)

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


def generate_decision(query, docs, llm=None, max_context_chars=4000):
    llm = llm or create_llm()
    context = _build_context(docs, max_context_chars)

    prompt = f"""
        Ты научный ассистент для датасета PubMedQA.

        Ответь только одним словом: yes, no или maybe.
        Используй только контекст ниже.

        Вопрос:
        {query}

        Контекст:
        {context}

        Ответ:
        """

    raw_answer = llm.invoke(prompt).strip().lower()
    for decision in VALID_DECISIONS:
        if raw_answer.startswith(decision):
            return decision, raw_answer

    return "unknown", raw_answer
