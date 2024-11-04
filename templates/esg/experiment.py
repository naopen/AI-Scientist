import argparse
import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import (
    HuggingFaceEmbeddings,
)  # 修正: langchain_huggingfaceから直接インポート
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
import gc

# 環境設定の最適化
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def clear_gpu_memory():
    """GPU メモリのクリーンアップ"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            with torch.cuda.device(f"cuda:{i}"):
                torch.cuda.empty_cache()
        gc.collect()


class SearchMode(Enum):
    FAISS = "faiss"
    GRAPH = "graph"
    HYBRID = "hybrid"


class GraphNode:
    def __init__(self, id: str, type: str, properties: Dict):
        self.id = id
        self.type = type
        self.properties = properties


class ExperimentConfig:
    def __init__(
        self,
        search_mode: SearchMode,
        split_mode: str,
        graph_mode: str,
        model_name: str = "stockmark/stockmark-100b-instruct-v0.1",
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        batch_size: int = 4,
        use_8bit: bool = False,
        use_float16: bool = True,
    ):
        self.search_mode = search_mode
        self.split_mode = split_mode
        self.graph_mode = graph_mode
        self.model_name = model_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.use_8bit = use_8bit
        self.use_float16 = use_float16


class MemoryOptimizedESGAnalyzer:
    def __init__(self, config: ExperimentConfig, experiment_dir: str):
        self.config = config
        self.experiment_dir = experiment_dir
        self.setup_device()
        clear_gpu_memory()
        self.setup_embeddings()
        self.setup_llm()
        self.load_data()

    def setup_device(self):
        """マルチGPUセットアップの最適化（修正版）"""
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            print("Warning: No GPU available. Using CPU.")
            return

        self.num_gpus = torch.cuda.device_count()
        print(f"\n=== GPU情報 ===")
        print(f"利用可能なGPU数: {self.num_gpus}")

        # GPUメモリの事前解放
        for i in range(self.num_gpus):
            with torch.cuda.device(f"cuda:{i}"):
                torch.cuda.empty_cache()

        total_memory = 0
        available_memory = 0
        for i in range(self.num_gpus):
            props = torch.cuda.get_device_properties(i)
            free_mem = torch.cuda.mem_get_info(i)[0]  # 実際の空きメモリを取得
            total_mem = props.total_memory
            total_memory += total_mem
            available_memory += free_mem
            print(
                f"GPU {i} ({props.name}): "
                f"総メモリ {total_mem/1e9:.2f}GB, "
                f"空きメモリ {free_mem/1e9:.2f}GB"
            )

        print(f"\n総GPU メモリ: {total_memory/1e9:.2f}GB")
        print(f"利用可能な総メモリ: {available_memory/1e9:.2f}GB")

        self.device = torch.device("cuda")

    def setup_embeddings(self):
        """埋め込みモデルの設定（さらなる修正版）"""
        try:
            gpu_id = self.num_gpus - 1

            print("\n=== Embedding Model Setup ===")
            print(f"Using GPU {gpu_id} for embeddings")

            # SentenceTransformerの初期化を修正
            self.sentence_transformer = SentenceTransformer(
                "intfloat/multilingual-e5-large"
            )
            self.sentence_transformer = self.sentence_transformer.to(f"cuda:{gpu_id}")

            if self.config.use_float16:
                self.sentence_transformer = self.sentence_transformer.half()

            # HuggingFaceEmbeddingsの初期化を修正（langchain_huggingfaceを使用）
            self.embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large",
                model_kwargs={"device": f"cuda:{gpu_id}"},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "device": f"cuda:{gpu_id}",
                },
            )

            if self.config.use_float16:
                # Embeddingsモデルを半精度に変換
                if hasattr(self.embeddings, "client") and hasattr(
                    self.embeddings.client, "model"
                ):
                    self.embeddings.client.model = self.embeddings.client.model.half()

            print("Embedding setup completed successfully")

        except Exception as e:
            print(f"Embedding setup error: {str(e)}")

    def setup_llm(self):
        """LLMの設定（デバイスマッピング修正版）"""
        try:
            print("\n=== LLM Setup ===")

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, legacy=False, trust_remote_code=True
            )

            # GPUメモリに基づいて最適なdevice_mapを構築
            if self.num_gpus > 1:
                # モデルの各レイヤーを自動的に分散
                device_map = "auto"
                max_memory = {
                    i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.85 / 1024**3)}GiB"
                    for i in range(self.num_gpus)
                }
                max_memory["cpu"] = "24GiB"  # CPU用のメモリ制限も設定
                print(
                    f"Using automatic device mapping with memory limits: {max_memory}"
                )
            else:
                device_map = "auto"
                max_memory = {0: "80GiB", "cpu": "24GiB"}
                print("Using single GPU mode with automatic mapping")

            # モデルの初期化パラメータ
            model_kwargs = {
                "device_map": device_map,
                "max_memory": max_memory,
                "trust_remote_code": True,
                "torch_dtype": (
                    torch.float16 if self.config.use_float16 else torch.float32
                ),
                "load_in_8bit": self.config.use_8bit,
            }

            # モデルのロードと最適化
            print(f"Loading model: {self.config.model_name}")
            model = AutoPeftModelForCausalLM.from_pretrained(
                self.config.model_name, **model_kwargs
            )

            # パイプラインの設定
            pipe_kwargs = {
                "model": model,
                "tokenizer": tokenizer,
                "max_new_tokens": 512,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.08,
                "return_full_text": False,
                "batch_size": self.config.batch_size * max(1, self.num_gpus),
            }

            if self.num_gpus > 1:
                pipe_kwargs["device_map"] = "auto"
            else:
                pipe_kwargs["device"] = 0

            pipe = pipeline("text-generation", **pipe_kwargs)

            self.llm = HuggingFacePipeline(pipeline=pipe)
            self.compressor = LLMChainExtractor.from_llm(self.llm)

            print("LLM setup completed successfully")

        except Exception as e:
            print(f"LLM setup error: {str(e)}")
            raise

    def load_data(self):
        """データ読み込み（エラー修正版）"""
        try:
            print("\n=== Loading Data ===")

            # FAISSベクトルストアの読み込み
            vectorstore_path = os.path.join(self.experiment_dir, "vectorstore")
            if not os.path.exists(vectorstore_path):
                raise ValueError(f"Vectorstore not found at {vectorstore_path}")

            print("Loading FAISS vectorstore...")
            self.vectorstore = FAISS.load_local(
                vectorstore_path, self.embeddings, allow_dangerous_deserialization=True
            )

            # メタデータの読み込み
            metadata_path = os.path.join(self.experiment_dir, "metadata.pkl")
            if not os.path.exists(metadata_path):
                print("Warning: metadata.pkl not found")
                self.metadata = {}
            else:
                print("Loading metadata...")
                with open(metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)

            # グラフデータの読み込み
            if self.config.graph_mode != "none":
                print("Loading graph data...")
                graph_pickle = os.path.join(self.experiment_dir, "graph.gpickle")
                graph_data = os.path.join(self.experiment_dir, "graph_data.pkl")

                if os.path.exists(graph_pickle):
                    self.graph = nx.read_gpickle(graph_pickle)
                    print(f"Loaded graph with {self.graph.number_of_nodes()} nodes")
                elif os.path.exists(graph_data):
                    with open(graph_data, "rb") as f:
                        data = pickle.load(f)
                        self.graph_data = {
                            "nodes": [
                                GraphNode(n["id"], n["type"], n["properties"])
                                for n in data["nodes"]
                            ],
                            "relations": data["relations"],
                        }
                    print(
                        f"Loaded graph data with {len(self.graph_data['nodes'])} nodes"
                    )
                else:
                    print("Warning: No graph data found")
                    self.graph = None
                    self.graph_data = None

            print("Data loading completed")

        except Exception as e:
            print(f"Data loading error: {str(e)}")
            raise

    def search_faiss(self, query: str) -> List[Dict]:
        """FAISSによる検索（メモリ最適化版）"""
        try:
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.top_k}
            )
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor, base_retriever=retriever
            )

            documents = compression_retriever.get_relevant_documents(query)
            results = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "search_type": "faiss",
                }
                for doc in documents
            ]

            return results

        except Exception as e:
            print(f"FAISS search error: {str(e)}")
            return []

    def search_graph(self, query: str) -> List[Dict]:
        """グラフベースの検索（メモリ最適化版）"""
        if not (self.graph or self.graph_data):
            return []

        try:
            # クエリの埋め込みを計算
            query_embedding = self.embeddings.embed_query(query)
            results = []

            if hasattr(self, "graph"):
                # NetworkXグラフを使用した検索
                for node in self.graph.nodes(data=True):
                    if node[1].get("type") == "Evidence":
                        content = node[1].get("content", "")
                        with torch.cuda.amp.autocast():
                            doc_embedding = self.embeddings.embed_query(content)
                            similarity = float(
                                cosine_similarity([query_embedding], [doc_embedding])[
                                    0
                                ][0]
                            )

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
                # カスタムグラフデータを使用した検索
                for node in self.graph_data["nodes"]:
                    if node.type == "Evidence":
                        content = node.properties.get("content", "")
                        with torch.cuda.amp.autocast():
                            doc_embedding = self.embeddings.embed_query(content)
                            similarity = float(
                                cosine_similarity([query_embedding], [doc_embedding])[
                                    0
                                ][0]
                            )

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

            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[: self.config.top_k]

        except Exception as e:
            print(f"Graph search error: {str(e)}")
            return []

    def search(self, query: str) -> List[Dict]:
        """検索の実行（ハイブリッドモード対応）"""
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

    def analyze_evidence(self, evidence: Dict) -> Dict:
        """根拠の分析（GPU最適化版）続き"""
        try:
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
            with torch.cuda.amp.autocast():
                response = self.llm(prompt)

            # レスポンスの解析
            try:
                scores = {}
                for line in response.split("\n"):
                    if "評価" in line and ":" in line:
                        key, value = line.split(":")
                        try:
                            # 数値のみを抽出
                            value = "".join(
                                filter(
                                    lambda x: x.isdigit() or x == ".",
                                    value.strip().split()[0],
                                )
                            )
                            scores[key.strip()] = float(value)
                        except:
                            continue

                comment = ""
                if "コメント:" in response:
                    comment = response.split("コメント:")[1].strip()

                return {"scores": scores, "comment": comment, "raw_response": response}
            except Exception as e:
                return {"error": str(e), "raw_response": response}

        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return {"error": str(e)}

    def analyze_evidence_batch(self, evidences: List[Dict]) -> List[Dict]:
        """根拠のバッチ分析（GPU最適化版）"""
        results = []
        batch_size = self.config.batch_size * max(1, self.num_gpus)

        for i in range(0, len(evidences), batch_size):
            batch = evidences[i : i + batch_size]
            batch_results = []

            for evidence in batch:
                result = self.analyze_evidence(evidence)
                batch_results.append(result)

            results.extend(batch_results)
            clear_gpu_memory()

        return results

    def find_related_evidences(self, evidence: Dict) -> List[Dict]:
        """関連する根拠の検索（GPU最適化版）"""
        try:
            if self.config.graph_mode != "none" and (self.graph or self.graph_data):
                related = []

                # NetworkXグラフを使用した関連検索
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

                # カスタムグラフデータを使用した関連検索
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
                            if target_node and target_node.type == "Evidence":
                                related.append(
                                    {
                                        "content": target_node.properties.get(
                                            "content", ""
                                        ),
                                        "metadata": target_node.properties,
                                        "relation_type": relation.type,
                                    }
                                )

                return related
            else:
                # FAISSベースの類似度検索
                query_embedding = self.embeddings.embed_query(evidence["content"])
                with torch.cuda.amp.autocast():
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

        except Exception as e:
            print(f"Related evidence search error: {str(e)}")
            return []

    def analyze_and_expand(self, query: str) -> Dict:
        """検索、分析、展開の実行（GPU最適化版）"""
        try:
            # 初期検索
            search_results = self.search(query)
            if not search_results:
                return {
                    "query": query,
                    "results": [],
                    "metadata": {
                        "total_results": 0,
                        "search_mode": self.config.search_mode.value,
                        "split_mode": self.config.split_mode,
                        "graph_mode": self.config.graph_mode,
                    },
                    "error": "No search results found",
                }

            # バッチ処理による分析
            analyzed_results = []
            batch_size = self.config.batch_size * max(1, self.num_gpus)

            for i in range(0, len(search_results), batch_size):
                batch = search_results[i : i + batch_size]
                batch_analyzed = []

                for result in batch:
                    analysis = self.analyze_evidence(result)
                    related_evidences = self.find_related_evidences(result)

                    batch_analyzed.append(
                        {
                            "evidence": result,
                            "analysis": analysis,
                            "related_evidences": related_evidences,
                        }
                    )

                analyzed_results.extend(batch_analyzed)
                clear_gpu_memory()

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

        except Exception as e:
            print(f"Analysis and expansion error: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "metadata": {
                    "search_mode": self.config.search_mode.value,
                    "split_mode": self.config.split_mode,
                    "graph_mode": self.config.graph_mode,
                },
            }

    def cleanup(self):
        """メモリの解放（マルチGPU対応）"""
        try:
            if hasattr(self, "embeddings"):
                del self.embeddings
            if hasattr(self, "sentence_transformer"):
                del self.sentence_transformer
            if hasattr(self, "llm"):
                if hasattr(self.llm, "pipeline"):
                    del self.llm.pipeline
                del self.llm
            if hasattr(self, "compressor"):
                del self.compressor
            if hasattr(self, "vectorstore"):
                del self.vectorstore
            if hasattr(self, "graph"):
                del self.graph
            if hasattr(self, "graph_data"):
                del self.graph_data

            clear_gpu_memory()
            print("Cleanup completed successfully")

        except Exception as e:
            print(f"Cleanup error: {str(e)}")


def run_experiment(config: ExperimentConfig, data_dir: str, output_dir: str):
    """実験の実行（マルチGPU対応）"""
    print(f"\n=== 実験開始: {config.search_mode.value} mode ===")
    start_time = time.time()

    try:
        experiment_dir = os.path.join(
            data_dir, f"experiment_{config.split_mode}_{config.graph_mode}"
        )

        if not os.path.exists(experiment_dir):
            raise ValueError(f"Experiment directory not found: {experiment_dir}")

        # アナライザーの初期化
        analyzer = MemoryOptimizedESGAnalyzer(config, experiment_dir)

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
            clear_gpu_memory()

        # アナライザーのクリーンアップ
        analyzer.cleanup()

        # 結果の保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(
            output_dir,
            f"experiment_results_{config.search_mode.value}_{timestamp}.json",
        )

        experiment_results = {
            "config": {
                "search_mode": config.search_mode.value,
                "split_mode": config.split_mode,
                "graph_mode": config.graph_mode,
                "model_name": config.model_name,
                "top_k": config.top_k,
                "similarity_threshold": config.similarity_threshold,
                "batch_size": config.batch_size,
                "use_8bit": config.use_8bit,
                "use_float16": config.use_float16,
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

    except Exception as e:
        print(f"Experiment error: {str(e)}")
        return None


def main(args):
    """メイン実行関数"""
    print("ESG分析実験を開始します...")

    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)

    # データディレクトリの確認
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory not found: {args.data_dir}")

    # 実験設定の組み合わせ
    experiment_configs = [
        # FAISSのみ（チャンク分割）
        ExperimentConfig(
            search_mode=SearchMode.FAISS,
            split_mode="chunk",
            graph_mode="none",
            batch_size=4,
            use_float16=True,
        ),
        # FAISSのみ（文単位分割）
        ExperimentConfig(
            search_mode=SearchMode.FAISS,
            split_mode="sentence",
            graph_mode="none",
            batch_size=4,
            use_float16=True,
        ),
        # グラフのみ（チャンク分割）
        ExperimentConfig(
            search_mode=SearchMode.GRAPH,
            split_mode="chunk",
            graph_mode="networkx",
            batch_size=4,
            use_float16=True,
        ),
        # グラフのみ（文単位分割）
        ExperimentConfig(
            search_mode=SearchMode.GRAPH,
            split_mode="sentence",
            graph_mode="networkx",
            batch_size=4,
            use_float16=True,
        ),
        # ハイブリッド（チャンク分割）
        ExperimentConfig(
            search_mode=SearchMode.HYBRID,
            split_mode="chunk",
            graph_mode="networkx",
            batch_size=4,
            use_float16=True,
        ),
        # ハイブリッド（文単位分割）
        ExperimentConfig(
            search_mode=SearchMode.HYBRID,
            split_mode="sentence",
            graph_mode="networkx",
            batch_size=4,
            use_float16=True,
        ),
    ]

    # 各設定で実験を実行
    all_results = {}
    successful_experiments = 0
    failed_experiments = 0

    for config in experiment_configs:
        try:
            print(f"\n実験設定:")
            print(f"- 検索モード: {config.search_mode.value}")
            print(f"- 分割モード: {config.split_mode}")
            print(f"- グラフモード: {config.graph_mode}")
            print(f"- バッチサイズ: {config.batch_size}")
            print(f"- Float16使用: {config.use_float16}")

            results = run_experiment(config, args.data_dir, args.output_dir)
            if results:
                all_results[f"{config.search_mode.value}_{config.split_mode}"] = results
                successful_experiments += 1

                # 個別の実験結果も保存
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                individual_result_file = os.path.join(
                    args.output_dir,
                    f"experiment_{config.search_mode.value}_{config.split_mode}_{timestamp}.json",
                )
                with open(individual_result_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"個別の実験結果を保存しました: {individual_result_file}")

        except Exception as e:
            print(
                f"実験エラー ({config.search_mode.value}_{config.split_mode}): {str(e)}"
            )
            failed_experiments += 1
            continue

        clear_gpu_memory()
        print(f"\nメモリクリーンアップ完了")

    # 実験サマリーの表示と保存
    print("\n=== 実験サマリー ===")
    print(f"成功した実験: {successful_experiments}")
    print(f"失敗した実験: {failed_experiments}")
    print(f"総実験数: {len(experiment_configs)}")

    # 結果が存在する場合のみ比較ファイルを作成
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = os.path.join(
            args.output_dir, f"experiment_comparison_{timestamp}.json"
        )

        try:
            with open(comparison_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"\n実験比較結果を保存しました: {comparison_file}")
        except Exception as e:
            print(f"\n警告: 比較結果の保存中にエラーが発生しました: {str(e)}")
    else:
        print("\n警告: 有効な実験結果がありませんでした。")

    print("\n実験プロセスが完了しました。")


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
        default="/root_nas05/home/2022/naoki/AI-Scientist/data/esg/experiments",
        help="Output directory for experiment results",
    )

    args = parser.parse_args()
    main(args)
