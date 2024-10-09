import matplotlib.pyplot as plt
import json
import os


def load_results(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    return results


def plot_topic_coverage(results, output_dir):
    topics = list(results["topic_coverage"].keys())
    coverage_scores = list(results["topic_coverage"].values())

    plt.figure(figsize=(12, 6))
    plt.bar(topics, coverage_scores)
    plt.title("ESGトピックのカバレッジ")
    plt.xlabel("トピック")
    plt.ylabel("カバレッジスコア")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "topic_coverage.png"))
    plt.close()


def plot_iteration_quality(results, output_dir):
    iterations = range(1, len(results["iteration_quality"]) + 1)
    quality_scores = results["iteration_quality"]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, quality_scores, marker="o")
    plt.title("イテレーションごとの抽出品質")
    plt.xlabel("イテレーション")
    plt.ylabel("品質スコア")
    plt.savefig(os.path.join(output_dir, "iteration_quality.png"))
    plt.close()


def main():
    input_file = "/root_nas05/home/2022/naoki/AI-Scientist/templates/esg/esg_analysis_results.json"
    output_dir = "/root_nas05/home/2022/naoki/AI-Scientist/templates/esg"

    results = load_results(input_file)
    plot_topic_coverage(results, output_dir)
    plot_iteration_quality(results, output_dir)


if __name__ == "__main__":
    main()
