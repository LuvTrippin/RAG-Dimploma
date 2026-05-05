import argparse
import json
import random
from pathlib import Path

from langchain_ollama import OllamaEmbeddings

from src.evaluation import (
    classification_accuracy,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    semantic_similarity,
)
from src.indexer import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    PERSIST_DIR,
    build_or_load_vectorstore,
)
from src.loader import load_pubmedqa
from src.pipeline import create_llm, generate_answer, generate_decision
from src.plot import plot_metrics
from src.retriever import get_retriever


def parse_args():
    parser = argparse.ArgumentParser(description="Run PubMedQA RAG evaluation.")
    parser.add_argument("--data-path", default="data/pubmedqa.json")
    parser.add_argument("--index-dir", default=PERSIST_DIR)
    parser.add_argument("--data-limit", type=int, default=1000)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--random-sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--llm-model", default="qwen:7b")
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument("--skip-answer-similarity", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--results-path", default="results/evaluation.json")
    parser.add_argument("--plots-dir", default="results/plots")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--show-sample", action="store_true")
    return parser.parse_args()


def select_samples(qa_pairs, sample_size, random_sample, seed):
    sample_size = min(sample_size, len(qa_pairs))
    if random_sample:
        rng = random.Random(seed)
        return rng.sample(qa_pairs, sample_size)
    return qa_pairs[:sample_size]


def average(results, key):
    values = [item[key] for item in results if item.get(key) is not None]
    if not values:
        return 0
    return sum(values) / len(values)


def save_results(path, payload):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()

    docs, qa_pairs = load_pubmedqa(args.data_path, limit=args.data_limit)

    vectorstore = build_or_load_vectorstore(
        docs,
        persist_dir=args.index_dir,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        rebuild=args.rebuild_index,
    )
    retriever = get_retriever(vectorstore, k=args.k)
    llm = create_llm(args.llm_model)
    eval_embeddings = OllamaEmbeddings(model=args.embedding_model)

    samples = select_samples(
        qa_pairs,
        sample_size=args.sample_size,
        random_sample=args.random_sample,
        seed=args.seed,
    )

    results = []
    for number, sample in enumerate(samples, start=1):
        query = sample["question"]
        qid = sample["id"]
        expected = sample["answer"]

        retrieved_docs = retriever.invoke(query)
        predicted, raw_prediction = generate_decision(
            query,
            retrieved_docs,
            llm=llm,
            max_context_chars=args.max_context_chars,
        )
        generated_answer = ""
        answer_similarity = None

        if not args.skip_answer_similarity:
            generated_answer = generate_answer(
                query,
                retrieved_docs,
                llm=llm,
                max_context_chars=args.max_context_chars,
            )
            answer_similarity = semantic_similarity(
                generated_answer,
                sample["long_answer"],
                embeddings=eval_embeddings,
            )

        result = {
            "number": number,
            "id": qid,
            "expected": expected,
            "predicted": predicted,
            "raw_prediction": raw_prediction,
            "generated_answer": generated_answer,
            "reference_answer": sample["long_answer"],
            "hit": hit_at_k(retrieved_docs, qid),
            "recall": recall_at_k(
                retrieved_docs,
                qid,
                relevant_contexts=sample["relevant_contexts"],
            ),
            "precision": precision_at_k(retrieved_docs, qid),
            "accuracy": classification_accuracy(predicted, expected),
            "answer_similarity": answer_similarity,
        }
        results.append(result)

    metrics = {
        f"hit@{args.k}": average(results, "hit"),
        f"recall@{args.k}": average(results, "recall"),
        f"precision@{args.k}": average(results, "precision"),
        "accuracy": average(results, "accuracy"),
    }
    if not args.skip_answer_similarity:
        metrics["answer_similarity"] = average(results, "answer_similarity")

    plot_paths = []
    if not args.skip_plots:
        try:
            plot_paths = plot_metrics(results, metrics, args.plots_dir)
        except RuntimeError as exc:
            print("Plot generation skipped:", exc)

    payload = {
        "config": vars(args),
        "metrics": metrics,
        "plots": plot_paths,
        "results": results,
    }
    save_results(args.results_path, payload)

    print("Done evaluation")
    print("Samples:", len(results))
    print("\n=== METRICS ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("Results saved to:", args.results_path)
    if plot_paths:
        print("Plots saved to:")
        for path in plot_paths:
            print("-", path)

    if args.show_sample and samples:
        sample = samples[0]
        retrieved_docs = retriever.invoke(sample["question"])
        answer = generate_answer(
            sample["question"],
            retrieved_docs,
            llm=llm,
            max_context_chars=args.max_context_chars,
        )

        print("\n=== SAMPLE TEST ===")
        print("QUESTION:", sample["question"])
        print("EXPECTED:", sample["answer"])
        print("ANSWER:", answer)


if __name__ == "__main__":
    main()
