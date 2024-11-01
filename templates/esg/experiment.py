import argparse
import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import pickle
import json
import time
import networkx as nx
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from enum import Enum

# Intel MKL設定
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


class SearchMode(Enum):
    FAISS = "faiss"
    GRAPH = "graph"
    HYBRID = "hybrid"


class ExperimentConfig:
    def __init__(
        self,
        search_mode: SearchMode,
        split_mode: str,
        graph_mode: str,
        model_name: str = "stockmark/stockmark-100b-instruct-v0.1",
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ):
        self.search_mode = search_mode
        self.split_mode = split_mode
        self.graph_mode = graph_mode
        self.model_name = model_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold


class ESGAnalyzer:
    def __init__(self, config: ExperimentConfig, experiment_dir: str):
        self.config = config
        self.experiment_dir = experiment_dir
        self.setup_embeddings()
        self.setup_llm()
        self.load_data()

    def setup_embeddings(self):
        """埋め込みモデルの設定"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large"
        )
        self.sentence_transformer = SentenceTransformer(
            "intfloat/multilingual-e5-large"
        )

    def setup_llm(self):
        """Stockmark-LLMの設定"""
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)
        self.compressor = LLMChainExtractor.from_llm(self.llm)

    def load_data(self):
        """実験データの読み込み"""
        # FAISSベクトルストアの読み込み
        self.vectorstore = FAISS.load_local(
            os.path.join(self.experiment_dir, "vectorstore"),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        # メタデータの読み込み
        with open(os.path.join(self.experiment_dir, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)

        # グラフデータの読み込み（存在する場合）
        if self.config.graph_mode != "none":
            if os.path.exists(os.path.join(self.experiment_dir, "graph.gpickle")):
                self.graph = nx.read_gpickle(
                    os.path.join(self.experiment_dir, "graph.gpickle")
                )
            elif os.path.exists(os.path.join(self.experiment_dir, "graph_data.pkl")):
                with open(
                    os.path.join(self.experiment_dir, "graph_data.pkl"), "rb"
                ) as f:
                    self.graph_data = pickle.load(f)
            else:
                self.graph = None
                self.graph_data = None

    def search_faiss(self, query: str) -> List[Dict]:
        """FAISSによる検索"""
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.top_k}
        )
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, base_retriever=retriever
        )

        documents = compression_retriever.get_relevant_documents(query)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "search_type": "faiss",
            }
            for doc in documents
        ]

    def search_graph(self, query: str) -> List[Dict]:
        """グラフベースの検索"""
        if not (self.graph or self.graph_data):
            return []

        # クエリの埋め込み
        query_embedding = self.embeddings.embed_query(query)

        results = []
        if hasattr(self, "graph"):
            # NetworkXグラフからの検索
            for node in self.graph.nodes(data=True):
                if node[1].get("type") == "Evidence":
                    content = node[1].get("content", "")
                    doc_embedding = self.embeddings.embed_query(content)
                    similarity = cosine_similarity([query_embedding], [doc_embedding])[
                        0
                    ][0]

                    if similarity > self.config.similarity_threshold:
                        results.append(
                            {
                                "content": content,
                                "metadata": node[1],
                                "similarity": similarity,
                                "search_type": "graph",
                                "node_id": node[0],
                            }
                        )

        elif hasattr(self, "graph_data"):
            # 保存されたグラフデータからの検索
            for node in self.graph_data["nodes"]:
                if node.type.value == "Evidence":
                    content = node.properties.get("content", "")
                    doc_embedding = self.embeddings.embed_query(content)
                    similarity = cosine_similarity([query_embedding], [doc_embedding])[
                        0
                    ][0]

                    if similarity > self.config.similarity_threshold:
                        results.append(
                            {
                                "content": content,
                                "metadata": node.properties,
                                "similarity": similarity,
                                "search_type": "graph",
                                "node_id": node.id,
                            }
                        )

        # 類似度でソートして上位k件を返す
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[: self.config.top_k]

    def analyze_evidence(self, evidence: Dict) -> Dict:
        """根拠の分析"""
        prompt = f"""
以下の文章から、ESG評価のための根拠としての質を分析してください。

文章: {evidence['content']}

分析の観点:
1. 具体性: 数値や具体的な事実が含まれているか
2. 検証可能性: 第三者が検証可能な情報か
3. 時期の明確性: いつの情報か明確か
4. 投資判断との関連性: 投資判断に影響を与える情報か

回答は以下の形式で書いてください:
- 具体性の評価（0-10点）:
- 検証可能性の評価（0-10点）:
- 時期の明確性の評価（0-10点）:
- 投資判断との関連性の評価（0-10点）:
- 総合評価（上記の平均点）:
- コメント:
"""
        response = self.llm(prompt)

        try:
            scores = {}
            for line in response.split("\n"):
                if "評価" in line and ":" in line:
                    key, value = line.split(":")
                    try:
                        scores[key.strip()] = float(value.strip().split()[0])
                    except:
                        continue

            comment = ""
            if "コメント:" in response:
                comment = response.split("コメント:")[1].strip()

            return {"scores": scores, "comment": comment, "raw_response": response}
        except Exception as e:
            return {"error": str(e), "raw_response": response}

    def find_related_evidences(self, evidence: Dict) -> List[Dict]:
        """関連する根拠の検索"""
        if self.config.graph_mode != "none" and (self.graph or self.graph_data):
            # グラフベースの関連性検索
            related = []
            if hasattr(self, "graph"):
                node_id = evidence.get("node_id")
                if node_id:
                    neighbors = self.graph.neighbors(node_id)
                    for neighbor in neighbors:
                        if self.graph.nodes[neighbor].get("type") == "Evidence":
                            related.append(
                                {
                                    "content": self.graph.nodes[neighbor].get(
                                        "content", ""
                                    ),
                                    "metadata": self.graph.nodes[neighbor],
                                    "relation_type": self.graph.edges[
                                        node_id, neighbor
                                    ].get("type", "unknown"),
                                }
                            )

            elif hasattr(self, "graph_data"):
                for relation in self.graph_data["relations"]:
                    if relation.source_id == evidence.get("node_id"):
                        target_node = next(
                            (
                                n
                                for n in self.graph_data["nodes"]
                                if n.id == relation.target_id
                            ),
                            None,
                        )
                        if target_node and target_node.type.value == "Evidence":
                            related.append(
                                {
                                    "content": target_node.properties.get(
                                        "content", ""
                                    ),
                                    "metadata": target_node.properties,
                                    "relation_type": relation.type.value,
                                }
                            )

            return related
        else:
            # FAISSベースの類似度検索
            query_embedding = self.embeddings.embed_query(evidence["content"])
            similar_docs = self.vectorstore.similarity_search_by_vector(
                query_embedding, k=self.config.top_k
            )
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "search_type": "similarity",
                }
                for doc in similar_docs
                if doc.page_content != evidence["content"]
            ]

    def search(self, query: str) -> List[Dict]:
        """検索の実行"""
        if self.config.search_mode == SearchMode.FAISS:
            return self.search_faiss(query)
        elif self.config.search_mode == SearchMode.GRAPH:
            return self.search_graph(query)
        else:  # HYBRID mode
            faiss_results = self.search_faiss(query)
            graph_results = self.search_graph(query)

            # 重複を除去して結果をマージ
            seen_contents = set()
            merged_results = []

            for result in faiss_results + graph_results:
                content = result["content"]
                if content not in seen_contents:
                    seen_contents.add(content)
                    merged_results.append(result)

            return merged_results[: self.config.top_k]

    def analyze_and_expand(self, query: str) -> Dict:
        """検索、分析、展開の実行"""
        # 初期検索
        search_results = self.search(query)

        # 各結果の分析
        analyzed_results = []
        for result in search_results:
            analysis = self.analyze_evidence(result)
            related_evidences = self.find_related_evidences(result)

            analyzed_results.append(
                {
                    "evidence": result,
                    "analysis": analysis,
                    "related_evidences": related_evidences,
                }
            )

        # 結果の評価とスコアリング
        scored_results = []
        for result in analyzed_results:
            if "scores" in result["analysis"]:
                total_score = sum(result["analysis"]["scores"].values()) / len(
                    result["analysis"]["scores"]
                )
                scored_results.append({**result, "total_score": total_score})

        # スコアで並び替え
        scored_results.sort(key=lambda x: x["total_score"], reverse=True)

        return {
            "query": query,
            "results": scored_results,
            "metadata": {
                "total_results": len(scored_results),
                "search_mode": self.config.search_mode.value,
                "split_mode": self.config.split_mode,
                "graph_mode": self.config.graph_mode,
            },
        }


def run_experiment(config: ExperimentConfig, data_dir: str, output_dir: str):
    """実験の実行"""
    print(f"\n=== 実験開始: {config.search_mode.value} mode ===")
    start_time = time.time()

    # 実験用ディレクトリの設定
    experiment_dir = os.path.join(
        data_dir, f"experiment_{config.split_mode}_{config.graph_mode}"
    )

    if not os.path.exists(experiment_dir):
        raise ValueError(f"Experiment directory not found: {experiment_dir}")

    # アナライザーの初期化
    analyzer = ESGAnalyzer(config, experiment_dir)

    # サンプルクエリの実行
    sample_queries = [
        "このレポートで議論されている主要なESG要因は何で、それらが投資判断にどのような影響を与える可能性がありますか？",
        "気候変動対策について、具体的な数値目標と実績を示してください。",
        "従業員の健康と安全に関する取り組みの実績を教えてください。",
        "コーポレートガバナンスの改善に向けた具体的な施策は何ですか？",
    ]

    results = []
    for query in tqdm(sample_queries, desc="Analyzing queries"):
        result = analyzer.analyze_and_expand(query)
        results.append(result)

    # 結果の保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(
        output_dir, f"experiment_results_{config.search_mode.value}_{timestamp}.json"
    )

    # 実験メタデータの追加
    experiment_results = {
        "config": {
            "search_mode": config.search_mode.value,
            "split_mode": config.split_mode,
            "graph_mode": config.graph_mode,
            "model_name": config.model_name,
            "top_k": config.top_k,
            "similarity_threshold": config.similarity_threshold,
        },
        "execution_time": time.time() - start_time,
        "timestamp": timestamp,
        "queries": results,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, ensure_ascii=False, indent=2)

    print(f"実験結果を保存しました: {output_filename}")
    print(f"実行時間: {experiment_results['execution_time']:.2f}秒")

    return experiment_results


def main(args):
    print("ESG分析実験を開始します...")

    # 実験設定の組み合わせ
    experiment_configs = [
        # FAISSのみ（チャンク分割）
        ExperimentConfig(
            search_mode=SearchMode.FAISS, split_mode="chunk", graph_mode="none"
        ),
        # FAISSのみ（文単位分割）
        ExperimentConfig(
            search_mode=SearchMode.FAISS, split_mode="sentence", graph_mode="none"
        ),
        # グラフのみ（チャンク分割）
        ExperimentConfig(
            search_mode=SearchMode.GRAPH, split_mode="chunk", graph_mode="networkx"
        ),
        # グラフのみ（文単位分割）
        ExperimentConfig(
            search_mode=SearchMode.GRAPH, split_mode="sentence", graph_mode="networkx"
        ),
        # ハイブリッド（チャンク分割）
        ExperimentConfig(
            search_mode=SearchMode.HYBRID, split_mode="chunk", graph_mode="networkx"
        ),
        # ハイブリッド（文単位分割）
        ExperimentConfig(
            search_mode=SearchMode.HYBRID, split_mode="sentence", graph_mode="networkx"
        ),
    ]

    # 各設定で実験を実行
    all_results = {}
    for config in experiment_configs:
        try:
            results = run_experiment(config, args.data_dir, args.output_dir)
            all_results[f"{config.search_mode.value}_{config.split_mode}"] = results
        except Exception as e:
            print(
                f"実験エラー ({config.search_mode.value}_{config.split_mode}): {str(e)}"
            )

    # 実験結果の比較分析
    comparison = analyze_experiment_results(all_results)

    # 比較結果の保存
    comparison_file = os.path.join(
        args.output_dir,
        f"experiment_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print(f"\n実験比較結果を保存しました: {comparison_file}")


def analyze_experiment_results(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """実験結果の比較分析"""
    comparison = {
        "execution_times": {},
        "average_scores": {},
        "query_performance": defaultdict(dict),
        "summary": {},
    }

    for exp_name, results in all_results.items():
        # 実行時間の比較
        comparison["execution_times"][exp_name] = results["execution_time"]

        # スコアの平均値計算
        scores = []
        for query_result in results["queries"]:
            for result in query_result["results"]:
                if "total_score" in result:
                    scores.append(result["total_score"])

        if scores:
            comparison["average_scores"][exp_name] = np.mean(scores)

        # クエリごとのパフォーマンス分析
        for query_idx, query_result in enumerate(results["queries"]):
            comparison["query_performance"][f"query_{query_idx}"][exp_name] = {
                "num_results": len(query_result["results"]),
                "avg_score": (
                    np.mean(
                        [
                            r["total_score"]
                            for r in query_result["results"]
                            if "total_score" in r
                        ]
                    )
                    if query_result["results"]
                    else 0
                ),
            }

    # 総合評価
    best_time = min(comparison["execution_times"].items(), key=lambda x: x[1])
    best_score = max(comparison["average_scores"].items(), key=lambda x: x[1])

    comparison["summary"] = {
        "fastest_method": {"name": best_time[0], "time": best_time[1]},
        "best_scoring_method": {"name": best_score[0], "score": best_score[1]},
        "recommendations": {
            "speed_focused": best_time[0],
            "quality_focused": best_score[0],
            "balanced": (
                best_score[0]
                if best_score[1] / max(comparison["average_scores"].values()) > 0.9
                else best_time[0]
            ),
        },
    }

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ESG analysis experiments")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/root_nas05/home/2022/naoki/AI-Scientist/data/esg/processed",
        help="Processed data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiment_results",
        help="Output directory for experiment results",
    )

    args = parser.parse_args()
    main(args)
