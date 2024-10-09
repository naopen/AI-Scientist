import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import os
import os.path as osp

# ESGカテゴリとそれに関連するキーワード
esg_categories = {
    "Environmental": [
        "気候変動対策",
        "再生可能エネルギー",
        "廃棄物管理",
        "水資源管理",
        "エネルギー効率",
        "生物多様性",
    ],
    "Social": [
        "労働条件",
        "人権",
        "ダイバーシティ",
        "従業員の健康と安全",
        "サプライチェーン管理",
        "地域社会への貢献",
    ],
    "Governance": [
        "コーポレートガバナンス",
        "リスク管理",
        "コンプライアンス",
        "取締役会の構成",
        "株主の権利",
        "経営の透明性",
    ],
}


# 結果の読み込み
def load_results(results_dir):
    datasets = ["esg"]
    folders = [
        f
        for f in os.listdir(results_dir)
        if f.startswith("run") and osp.isdir(osp.join(results_dir, f))
    ]
    all_results = {}

    for folder in folders:
        folder_path = osp.join(results_dir, folder)
        with open(osp.join(folder_path, "all_results.json"), "r") as f:
            results = json.load(f)

        for dataset in datasets:
            if dataset not in all_results:
                all_results[dataset] = {}

            for key, value in results.items():
                if dataset in key:
                    seed = key.split("_")[-1]
                    if seed not in all_results[dataset]:
                        all_results[dataset][seed] = []
                    all_results[dataset][seed].append(value)

    return all_results


# トレーニング損失と検証損失のプロット
def plot_losses(results, dataset, out_dir):
    plt.figure(figsize=(10, 6))
    for seed, runs in results[dataset].items():
        for i, run in enumerate(runs):
            plt.plot(run["train_losses"], label=f"Train (Seed {seed}, Run {i+1})")
            plt.plot(run["val_losses"], label=f"Val (Seed {seed}, Run {i+1})")

    plt.title(f"Training and Validation Losses for {dataset.upper()}")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(osp.join(out_dir, f"{dataset}_losses.png"))
    plt.close()


# ESGカテゴリごとの性能プロット
def plot_esg_performance(results, dataset, out_dir):
    categories = list(esg_categories.keys())

    plt.figure(figsize=(12, 6))
    x = np.arange(len(categories))
    width = 0.35

    for seed, runs in results[dataset].items():
        performances = [
            np.mean([run["esg_performance"][cat] for run in runs]) for cat in categories
        ]
        plt.bar(
            x + float(seed) * width / len(results[dataset]),
            performances,
            width / len(results[dataset]),
            label=f"Seed {seed}",
        )

    plt.title(f"ESG Category Performance for {dataset.upper()}")
    plt.xlabel("ESG Categories")
    plt.ylabel("Performance Score")
    plt.xticks(x + width / 2, categories)
    plt.legend()
    plt.savefig(osp.join(out_dir, f"{dataset}_esg_performance.png"))
    plt.close()


# キーワード抽出性能のヒートマップ
def plot_keyword_heatmap(results, dataset, out_dir):
    all_keywords = [kw for cat in esg_categories.values() for kw in cat]

    keyword_scores = {kw: [] for kw in all_keywords}
    for seed, runs in results[dataset].items():
        for run in runs:
            for kw, score in run["keyword_extraction"].items():
                keyword_scores[kw].append(score)

    avg_scores = np.array(
        [np.mean(scores) for scores in keyword_scores.values()]
    ).reshape(3, 4)

    plt.figure(figsize=(12, 8))
    plt.imshow(avg_scores, cmap="YlOrRd", aspect="auto")
    plt.title(f"Keyword Extraction Performance for {dataset.upper()}")
    plt.xlabel("Keywords")
    plt.ylabel("ESG Categories")
    plt.xticks(
        np.arange(4),
        ["Keyword 1", "Keyword 2", "Keyword 3", "Keyword 4"],
        rotation=45,
        ha="right",
    )
    plt.yticks(np.arange(3), list(esg_categories.keys()))

    for i in range(3):
        for j in range(4):
            plt.text(j, i, f"{avg_scores[i, j]:.2f}", ha="center", va="center")

    plt.colorbar(label="Extraction Score")
    plt.tight_layout()
    plt.savefig(osp.join(out_dir, f"{dataset}_keyword_heatmap.png"))
    plt.close()


# 生成されたサンプルテキストの表示
def display_generated_samples(results, dataset, out_dir):
    with open(osp.join(out_dir, f"{dataset}_generated_samples.txt"), "w") as f:
        for seed, runs in results[dataset].items():
            f.write(f"Seed {seed}:\n")
            for i, run in enumerate(runs):
                f.write(f"  Run {i+1}:\n")
                for j, sample in enumerate(
                    run["generated_samples"][:3]
                ):  # 最初の3つのサンプルのみ表示
                    f.write(
                        f"    Sample {j+1}: {sample[:200]}...\n"
                    )  # 最初の200文字のみ表示
                f.write("\n")
            f.write("\n")


def main():
    # experiment.pyと同じディレクトリを使用
    current_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = current_dir

    all_results = load_results(current_dir)

    for dataset in all_results.keys():
        plot_losses(all_results, dataset, out_dir)
        plot_esg_performance(all_results, dataset, out_dir)
        plot_keyword_heatmap(all_results, dataset, out_dir)
        display_generated_samples(all_results, dataset, out_dir)

    print(f"Plots and analysis have been saved to {out_dir}")


if __name__ == "__main__":
    main()
