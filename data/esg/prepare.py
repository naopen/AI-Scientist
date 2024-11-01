from pathlib import Path
import torch
from tqdm import tqdm
import pickle
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import functools
from ja_sentence_segmenter.common.pipeline import make_pipeline
from ja_sentence_segmenter.concatenate.simple_concatenator import concatenate_matching
from ja_sentence_segmenter.normalize.neologd_normalizer import normalize
from ja_sentence_segmenter.split.simple_splitter import split_newline, split_punctuation
import spacy
import re
from datetime import datetime
from neo4j import GraphDatabase
import networkx as nx
from typing import NamedTuple


class SplitMode(Enum):
    CHUNK = "chunk"
    SENTENCE = "sentence"


class GraphMode(Enum):
    NONE = "none"  # グラフを使用しない
    NEO4J = "neo4j"  # Neo4jを使用
    NETWORKX = "networkx"  # NetworkXを使用（メモリ内グラフ）


class NodeType(Enum):
    METRIC = "Metric"  # 指標
    VALUE = "Value"  # 数値
    ACTION = "Action"  # 施策
    TIME = "Time"  # 時期
    EVIDENCE = "Evidence"  # 根拠文


class RelationType(Enum):
    HAS_VALUE = "HAS_VALUE"  # 指標-数値間の関係
    LEADS_TO = "LEADS_TO"  # 施策-指標間の関係
    MEASURED_AT = "MEASURED_AT"  # 値-時期間の関係
    CONTRIBUTES_TO = "CONTRIBUTES_TO"  # 施策間の貢献関係
    REFERS_TO = "REFERS_TO"  # 根拠文-他ノード間の参照関係


@dataclass
class ProcessingConfig:
    split_mode: SplitMode
    graph_mode: GraphMode
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_sentences: int = 3
    sentence_overlap: int = 1
    num_examples: int = 5
    start_index: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"


class GraphNode(NamedTuple):
    id: str
    type: NodeType
    properties: Dict


class GraphRelation(NamedTuple):
    source_id: str
    target_id: str
    type: RelationType
    properties: Dict


class GraphBuilder:
    def __init__(self, mode: GraphMode, config: ProcessingConfig):
        self.mode = mode
        self.config = config
        self.nlp = spacy.load("ja_core_news_lg")

        if mode == GraphMode.NEO4J:
            self.driver = GraphDatabase.driver(
                config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password)
            )
        elif mode == GraphMode.NETWORKX:
            self.graph = nx.MultiDiGraph()

    def extract_metrics(self, text: str) -> List[Tuple[str, float]]:
        metrics = []
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["PERCENT", "QUANTITY", "MONEY"]:
                value = re.findall(r"[\d.]+", ent.text)
                if value:
                    metrics.append((ent.text, float(value[0])))
        return metrics

    def extract_time_references(self, text: str) -> List[str]:
        time_refs = []
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["DATE", "TIME"]:
                time_refs.append(ent.text)
        return time_refs

    def extract_actions(self, text: str) -> List[str]:
        actions = []
        doc = self.nlp(text)
        for sent in doc.sents:
            if any(word in sent.text for word in ["実施", "導入", "推進", "取り組み"]):
                actions.append(sent.text)
        return actions

    def build_graph_from_document(
        self, doc: Document
    ) -> Tuple[List[GraphNode], List[GraphRelation]]:
        nodes = []
        relations = []
        text = doc.page_content

        evidence_id = f"evidence_{len(nodes)}"
        evidence_node = GraphNode(
            id=evidence_id,
            type=NodeType.EVIDENCE,
            properties={"content": text, **doc.metadata},
        )
        nodes.append(evidence_node)

        metrics = self.extract_metrics(text)
        for metric_text, value in metrics:
            metric_id = f"metric_{len(nodes)}"
            value_id = f"value_{len(nodes)}"

            nodes.append(
                GraphNode(
                    id=metric_id, type=NodeType.METRIC, properties={"name": metric_text}
                )
            )

            nodes.append(
                GraphNode(id=value_id, type=NodeType.VALUE, properties={"value": value})
            )

            relations.extend(
                [
                    GraphRelation(metric_id, value_id, RelationType.HAS_VALUE, {}),
                    GraphRelation(evidence_id, metric_id, RelationType.REFERS_TO, {}),
                ]
            )

        time_refs = self.extract_time_references(text)
        for time_ref in time_refs:
            time_id = f"time_{len(nodes)}"
            nodes.append(
                GraphNode(id=time_id, type=NodeType.TIME, properties={"time": time_ref})
            )
            relations.append(
                GraphRelation(evidence_id, time_id, RelationType.MEASURED_AT, {})
            )

        actions = self.extract_actions(text)
        for action in actions:
            action_id = f"action_{len(nodes)}"
            nodes.append(
                GraphNode(
                    id=action_id,
                    type=NodeType.ACTION,
                    properties={"description": action},
                )
            )
            relations.append(
                GraphRelation(evidence_id, action_id, RelationType.REFERS_TO, {})
            )

        return nodes, relations

    def save_graph(self, output_path: Path):
        if self.mode == GraphMode.NETWORKX:
            import pickle

            # グラフをpickleとして保存
            with open(output_path / "graph.pkl", "wb") as f:
                pickle.dump(self.graph, f)

    @staticmethod
    def load_graph(path: Path):
        if path.exists():
            import pickle

            with open(path / "graph.pkl", "rb") as f:
                return pickle.load(f)
        return None

    def close(self):
        if self.mode == GraphMode.NEO4J:
            self.driver.close()


class CustomEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, device: str):
        super().__init__(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _add_prompt(self, texts: List[str], prompt_type: str) -> List[str]:
        if prompt_type == "query":
            return [f"query: {text}" for text in texts]
        elif prompt_type == "document":
            return [f"document: {text}" for text in texts]
        return texts

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = self._add_prompt(texts, "document")
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        text = self._add_prompt([text], "query")[0]
        return super().embed_query(text)


class ESGDocumentProcessor:
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig(
            split_mode=SplitMode.CHUNK, graph_mode=GraphMode.NONE
        )
        self.embeddings = CustomEmbeddings(device=self.config.device)

        if self.config.graph_mode != GraphMode.NONE:
            self.graph_builder = GraphBuilder(self.config.graph_mode, self.config)

        split_punc2 = functools.partial(split_punctuation, punctuations=r".。!?")
        concat_tail_no = functools.partial(
            concatenate_matching,
            former_matching_rule=r"^(?P<result>.+)(の)$",
            remove_former_matched=False,
        )
        concat_tail_te = functools.partial(
            concatenate_matching,
            former_matching_rule=r"^(?P<result>.+)(て)$",
            remove_former_matched=False,
        )
        concat_decimal = functools.partial(
            concatenate_matching,
            former_matching_rule=r"^(?P<result>.+)(\d.)$",
            latter_matching_rule=r"^(\d)(?P<result>.+)$",
            remove_former_matched=False,
            remove_latter_matched=False,
        )
        self.segmenter = make_pipeline(
            normalize,
            split_newline,
            concat_tail_no,
            concat_tail_te,
            split_punc2,
            concat_decimal,
        )

        self.setup_logging()

    @staticmethod
    def setup_logging():
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def split_into_sentences(self, text: str) -> List[str]:
        return list(self.segmenter(text))

    def create_sentence_windows(self, sentences: List[str]) -> List[str]:
        if not sentences:
            return []

        windows = []
        for i in range(
            0, len(sentences), self.config.max_sentences - self.config.sentence_overlap
        ):
            window = sentences[i : i + self.config.max_sentences]
            if window:
                windows.append(" ".join(window))
        return windows

    def display_examples(self, documents: List[Document], file_name: str):
        total_docs = len(documents)
        start_idx = min(self.config.start_index, total_docs)
        end_idx = min(start_idx + self.config.num_examples, total_docs)

        print(f"\n=== 分割例 ({file_name}) ===")
        print(f"分割モード: {self.config.split_mode.value}")
        print(f"グラフモード: {self.config.graph_mode.value}")
        print(f"総文書数: {total_docs}")
        print(f"表示範囲: {start_idx + 1}番目 ～ {end_idx}番目")

        if self.config.split_mode == SplitMode.CHUNK:
            print(f"チャンクサイズ: {self.config.chunk_size}")
            print(f"オーバーラップ: {self.config.chunk_overlap}")
        else:
            print(f"文数: {self.config.max_sentences}")
            print(f"文オーバーラップ: {self.config.sentence_overlap}")

        # print("\n=== 分割結果 ===")
        # for i, doc in enumerate(documents[start_idx:end_idx], start=start_idx + 1):
        #     print(f"\n例 {i}:")
        #     print(f"長さ: {len(doc.page_content)} 文字")
        #     print("内容:")
        #     print(doc.page_content)
        #     print("-" * 80)

    def process_document(
        self, file_path: Path
    ) -> Tuple[List[Document], Optional[Tuple[List[GraphNode], List[GraphRelation]]]]:
        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            processed_docs = []

            # 文書分割処理
            if self.config.split_mode == SplitMode.CHUNK:
                text_splitter = CharacterTextSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )
                processed_docs = text_splitter.split_documents(documents)
            else:  # SENTENCE mode
                for doc in documents:
                    sentences = self.split_into_sentences(doc.page_content)
                    windows = self.create_sentence_windows(sentences)

                    for window in windows:
                        processed_docs.append(
                            Document(
                                page_content=window,
                                metadata={
                                    **doc.metadata,
                                    "split_type": "sentence",
                                    "window_size": self.config.max_sentences,
                                },
                            )
                        )

            if self.config.num_examples > 0:
                self.display_examples(processed_docs, file_path.name)

            # グラフ構造の構築（必要な場合）
            if self.config.graph_mode != GraphMode.NONE:
                all_nodes = []
                all_relations = []
                for doc in processed_docs:
                    nodes, relations = self.graph_builder.build_graph_from_document(doc)
                    all_nodes.extend(nodes)
                    all_relations.extend(relations)
                return processed_docs, (all_nodes, all_relations)

            return processed_docs, None

        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            return [], None

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        return FAISS.from_documents(documents, self.embeddings)

    def process_directory(self, input_dir: Path, output_dir: Path):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_documents = []
        all_graph_data = (
            {"nodes": [], "relations": []}
            if self.config.graph_mode != GraphMode.NONE
            else None
        )

        pdf_files = list(input_dir.glob("*.pdf"))

        if not pdf_files:
            logging.error(f"No PDF files found in {input_dir}")
            return

        for pdf_file in tqdm(pdf_files, desc="Processing PDF files"):
            try:
                docs, graph_data = self.process_document(pdf_file)
                if docs:
                    all_documents.extend(docs)
                    if graph_data:
                        nodes, relations = graph_data
                        all_graph_data["nodes"].extend(nodes)
                        all_graph_data["relations"].extend(relations)
                    logging.info(f"Successfully processed {pdf_file.name}")
                else:
                    logging.warning(f"No documents extracted from {pdf_file.name}")
            except Exception as e:
                logging.error(f"Error processing {pdf_file}: {str(e)}")

        if not all_documents:
            logging.error("No documents were successfully processed")
            return

        logging.info(f"Total documents processed: {len(all_documents)}")

        try:
            # FAISSベクトルストアの作成と保存
            vector_store = self.create_vector_store(all_documents)
            vector_store.save_local(str(output_dir / "vectorstore"))

            # メタデータの保存
            meta_data = {
                "total_documents": len(all_documents),
                "embedding_model": "intfloat/multilingual-e5-large",
                "split_mode": self.config.split_mode.value,
                "graph_mode": self.config.graph_mode.value,
                "processing_config": {
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap,
                    "max_sentences": self.config.max_sentences,
                    "sentence_overlap": self.config.sentence_overlap,
                },
                "processing_timestamp": datetime.now().isoformat(),
            }

            if all_graph_data:
                meta_data["total_nodes"] = len(all_graph_data["nodes"])
                meta_data["total_relations"] = len(all_graph_data["relations"])
                meta_data["node_type_distribution"] = {
                    node_type.value: len(
                        [n for n in all_graph_data["nodes"] if n.type == node_type]
                    )
                    for node_type in NodeType
                }

            with open(output_dir / "metadata.pkl", "wb") as f:
                pickle.dump(meta_data, f)

            # グラフデータの保存
            if all_graph_data:
                if self.config.graph_mode == GraphMode.NETWORKX:
                    self.graph_builder.save_graph(output_dir)

                # グラフデータの詳細情報も保存
                with open(output_dir / "graph_data.pkl", "wb") as f:
                    pickle.dump(all_graph_data, f)

            logging.info("Successfully saved processed data")

        except Exception as e:
            logging.error(f"Error saving processed data: {str(e)}")

        finally:
            if self.config.graph_mode != GraphMode.NONE:
                self.graph_builder.close()


def main():
    base_input_dir = Path("/root_nas05/home/2022/naoki/AI-Scientist/data/esg")
    base_output_dir = Path(
        "/root_nas05/home/2022/naoki/AI-Scientist/data/esg/processed"
    )

    # 実験設定の組み合わせ
    configs = [
        # チャンク分割 + FAISSのみ
        ProcessingConfig(
            split_mode=SplitMode.CHUNK,
            graph_mode=GraphMode.NONE,
            chunk_size=500,
            chunk_overlap=50,
        ),
        # チャンク分割 + FAISS + グラフ
        ProcessingConfig(
            split_mode=SplitMode.CHUNK,
            graph_mode=GraphMode.NETWORKX,
            chunk_size=500,
            chunk_overlap=50,
        ),
        # 文単位分割 + FAISSのみ
        ProcessingConfig(
            split_mode=SplitMode.SENTENCE,
            graph_mode=GraphMode.NONE,
            max_sentences=3,
            sentence_overlap=1,
        ),
        # 文単位分割 + FAISS + グラフ
        ProcessingConfig(
            split_mode=SplitMode.SENTENCE,
            graph_mode=GraphMode.NETWORKX,
            max_sentences=3,
            sentence_overlap=1,
        ),
    ]

    # 各設定で処理を実行
    for i, config in enumerate(configs):
        print(f"\n=== 実験 {i+1}/{len(configs)} ===")
        print(f"分割モード: {config.split_mode.value}")
        print(f"グラフモード: {config.graph_mode.value}")

        # 出力ディレクトリの設定
        experiment_dir = (
            base_output_dir
            / f"experiment_{config.split_mode.value}_{config.graph_mode.value}"
        )
        processor = ESGDocumentProcessor(config)

        try:
            processor.process_directory(base_input_dir, experiment_dir)
            print(f"実験 {i+1} 完了: {experiment_dir}")
        except Exception as e:
            print(f"実験 {i+1} エラー: {str(e)}")


if __name__ == "__main__":
    main()
