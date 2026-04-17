import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intent import classify
from tools.chat import general_chat
from tools.summarizer import summarize_text


DEFAULT_DATASET = Path(__file__).parent / "intent_samples.json"


def _normalize_intents(items: list[dict]) -> set[str]:
    return {item.get("intent", "general_chat") for item in items}


def benchmark_intent_classification(dataset_path: Path) -> dict:
    samples = json.loads(dataset_path.read_text(encoding="utf-8"))
    total = len(samples)
    if total == 0:
        raise ValueError("Benchmark dataset is empty.")

    exact_matches = 0
    latencies_ms = []
    details = []

    for sample in samples:
        text = sample["text"]
        expected = set(sample["expected_intents"])

        start = time.perf_counter()
        predicted_items = classify(text, chat_history=[])
        elapsed_ms = (time.perf_counter() - start) * 1000

        predicted = _normalize_intents(predicted_items)
        matched = predicted == expected
        if matched:
            exact_matches += 1

        latencies_ms.append(elapsed_ms)
        details.append(
            {
                "text": text,
                "expected": sorted(expected),
                "predicted": sorted(predicted),
                "matched": matched,
                "latency_ms": round(elapsed_ms, 2),
            }
        )

    return {
        "total_samples": total,
        "exact_match_accuracy": round(exact_matches / total, 4),
        "latency_ms": {
            "avg": round(statistics.mean(latencies_ms), 2),
            "p50": round(statistics.median(latencies_ms), 2),
            "max": round(max(latencies_ms), 2),
        },
        "samples": details,
    }


def benchmark_generation_latency(rounds: int) -> dict:
    chat_latencies = []
    summarize_latencies = []

    for _ in range(rounds):
        start = time.perf_counter()
        _ = general_chat("Give a two-line explanation of recursion.", chat_history=[])
        chat_latencies.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        _ = summarize_text(
            "Neural networks are computational models inspired by the brain. "
            "They are widely used in computer vision, NLP, and speech tasks."
        )
        summarize_latencies.append((time.perf_counter() - start) * 1000)

    return {
        "rounds": rounds,
        "chat_latency_ms": {
            "avg": round(statistics.mean(chat_latencies), 2),
            "p50": round(statistics.median(chat_latencies), 2),
            "max": round(max(chat_latencies), 2),
        },
        "summarize_latency_ms": {
            "avg": round(statistics.mean(summarize_latencies), 2),
            "p50": round(statistics.median(summarize_latencies), 2),
            "max": round(max(summarize_latencies), 2),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run model benchmarks for intent classification and generation latency."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the JSON benchmark dataset for intent classification.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="How many rounds to run for chat/summarization latency benchmarks.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional file path to save benchmark results as JSON.",
    )
    args = parser.parse_args()

    intent_results = benchmark_intent_classification(args.dataset)
    latency_results = benchmark_generation_latency(args.rounds)
    report = {
        "intent_classification": intent_results,
        "generation_latency": latency_results,
    }

    print(json.dumps(report, indent=2))

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved benchmark report to: {args.save}")


if __name__ == "__main__":
    main()
