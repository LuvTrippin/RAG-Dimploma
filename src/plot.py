from collections import Counter
from pathlib import Path


METRIC_LABELS = {
    "hit": "Hit",
    "recall": "Recall",
    "precision": "Precision",
    "accuracy": "Accuracy",
    "answer_similarity": "Answer similarity",
}


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to build plots. Install dependencies from "
            "requirements.txt or run with --skip-plots."
        ) from exc

    return plt


def _save_current_plot(plt, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def _numeric_values(results, key):
    return [item[key] for item in results if item.get(key) is not None]


def _plot_metric_histogram(results, key, output_dir):
    plt = _load_pyplot()
    values = _numeric_values(results, key)
    if not values:
        return None

    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=10, range=(0, 1), edgecolor="black", color="#4c78a8")
    plt.title(f"{METRIC_LABELS.get(key, key)} distribution")
    plt.xlabel("Score")
    plt.ylabel("Samples")
    plt.xlim(0, 1)
    plt.grid(axis="y", alpha=0.25)
    return _save_current_plot(plt, output_dir / f"{key}_histogram.png")


def _plot_metric_overview(metrics, output_dir):
    plt = _load_pyplot()
    if not metrics:
        return None

    names = list(metrics.keys())
    values = [metrics[name] for name in names]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(names, values, color="#59a14f")
    plt.title("Average evaluation metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.02, 0.98),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    return _save_current_plot(plt, output_dir / "metrics_overview.png")


def _plot_answer_distribution(results, output_dir):
    plt = _load_pyplot()
    expected = Counter(item.get("expected") for item in results if item.get("expected"))
    predicted = Counter(item.get("predicted") for item in results if item.get("predicted"))
    labels = sorted(set(expected) | set(predicted))
    if not labels:
        return None

    x_positions = range(len(labels))
    width = 0.38

    plt.figure(figsize=(7, 4))
    plt.bar(
        [x - width / 2 for x in x_positions],
        [expected[label] for label in labels],
        width=width,
        label="Expected",
        color="#4c78a8",
    )
    plt.bar(
        [x + width / 2 for x in x_positions],
        [predicted[label] for label in labels],
        width=width,
        label="Predicted",
        color="#f28e2b",
    )
    plt.title("Expected vs predicted answers")
    plt.xlabel("Answer")
    plt.ylabel("Samples")
    plt.xticks(list(x_positions), labels)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    return _save_current_plot(plt, output_dir / "answer_distribution.png")


def plot_metrics(results, metrics, output_dir):
    output_dir = Path(output_dir)
    plot_paths = []

    overview_path = _plot_metric_overview(metrics, output_dir)
    if overview_path:
        plot_paths.append(overview_path)

    for key in ("hit", "recall", "precision", "accuracy", "answer_similarity"):
        path = _plot_metric_histogram(results, key, output_dir)
        if path:
            plot_paths.append(path)

    answer_distribution_path = _plot_answer_distribution(results, output_dir)
    if answer_distribution_path:
        plot_paths.append(answer_distribution_path)

    return plot_paths
