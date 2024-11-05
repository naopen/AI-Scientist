import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import logging
from datetime import datetime
import pickle
import json
from tqdm import tqdm
import re
import numpy as np
from functools import partial
from cryptography.fernet import Fernet


@dataclass
class SearchConfig:
    """検索システムの設定"""

    chunk_size: int = 300
    chunk_overlap: int = 50
    model_name: str = "intfloat/multilingual-e5-large"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 16
    similarity_threshold: float = 0.2
    max_results: int = 30


@dataclass
class DocumentInfo:
    """文書情報を保持するクラス"""

    company: str
    year: int
    industry: str
    report_type: str
    filepath: Path

    @classmethod
    def from_path(cls, path: Path) -> "DocumentInfo":
        """パスから文書情報を抽出"""
        # 例: 自動車/アイシン_2023_integrated.pdf
        industry = path.parent.name
        filename_parts = path.stem.split("_")
        return cls(
            company=filename_parts[0],
            year=int(filename_parts[1]),
            industry=industry,
            report_type=filename_parts[2],
            filepath=path,
        )


class ESGTextPreprocessor:
    """テキストの前処理クラス"""

    @staticmethod
    def clean_text(text: str) -> str:
        """テキストのクリーニング"""
        # 余分な空白の削除
        text = re.sub(r"\s+", " ", text)
        # 特殊文字の正規化
        text = text.replace("〜", "～").replace("－", "-")
        # 括弧の正規化
        text = text.replace("（", "(").replace("）", ")")
        return text.strip()

    @staticmethod
    def is_valid_sentence(text: str) -> bool:
        """有効な文かどうかの判定"""
        if len(text) < 10:  # 短すぎる文を除外
            return False
        if not re.search(
            r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]", text
        ):
            return False
        # if len(re.findall(r"[。！？!?]", text)) > 3:  # 複数の文が混ざっている場合を除外
        #     return False
        if re.fullmatch(
            r"[\u3040-\u309f\u30a0-\u30ff]+", text
        ):  # ひらがな・カタカナのみの文を除外
            return False
        return True


class ESGDocumentProcessor:
    """文書処理クラス"""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.preprocessor = ESGTextPreprocessor()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "、", " "],
        )

    def process_pdf(self, pdf_path: Path) -> List[Dict]:
        """PDFの処理"""
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            processed_chunks = []

            for page in pages:
                # メタデータの準備
                metadata = {
                    "source": pdf_path.name,
                    "page": page.metadata.get("page", 0),
                    "company": self._extract_company_name(pdf_path.name),
                }

                # テキストの前処理
                cleaned_text = self.preprocessor.clean_text(page.page_content)

                # チャンクへの分割
                chunks = self.text_splitter.split_text(cleaned_text)

                # 各チャンクの処理
                for chunk in chunks:
                    if self.preprocessor.is_valid_sentence(chunk):
                        processed_chunks.append(
                            {"content": chunk, "metadata": metadata.copy()}
                        )

            return processed_chunks

        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {str(e)}")
            return []

    @staticmethod
    def _extract_company_name(filename: str) -> str:
        """ファイル名から企業名を抽出"""
        # 企業名の抽出ロジックを実装
        match = re.match(r"([^_]+)_\d{4}", filename)
        return match.group(1) if match else filename.split("_")[0]


class ESGSearchSystem:
    """検索システムのメインクラス"""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = SentenceTransformer(config.model_name)
        self.model.to(self.device)

        if torch.cuda.is_available():
            self.model = self.model.half()

        self.processor = ESGDocumentProcessor(config)
        self.vectorstore = None
        self.document_map = {}

    def encode_query(self, text: str) -> np.ndarray:
        """クエリテキストをエンコード"""
        with torch.amp.autocast(device_type=self.config.device):
            embedding = self.model.encode(
                text, convert_to_tensor=True, show_progress_bar=False
            )
        return embedding.cpu().numpy()

    def build_index(self, base_dir: Path, save_dir: Path):
        """検索インデックスの構築"""
        logging.info("Building search index...")
        all_chunks = []

        # 業界ディレクトリの処理
        industry_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        for industry_dir in industry_dirs:
            pdf_files = list(industry_dir.glob("*.pdf"))
            for pdf_path in tqdm(
                pdf_files, desc=f"Processing {industry_dir.name} PDFs"
            ):
                try:
                    doc_info = DocumentInfo.from_path(pdf_path)
                    chunks = self.processor.process_pdf(pdf_path)
                    for chunk in chunks:
                        chunk["metadata"]["industry"] = doc_info.industry
                    all_chunks.extend(chunks)

                    # ドキュメントマップの更新
                    for chunk in chunks:
                        self.document_map[chunk["content"]] = chunk["metadata"]
                except Exception as e:
                    logging.error(f"Error processing {pdf_path}: {str(e)}")
                    continue

        if not all_chunks:
            raise ValueError("No valid chunks extracted from PDFs")

        # ベクトルストアの構築
        texts = [chunk["content"] for chunk in all_chunks]
        embeddings = self._batch_encode(texts)
        embeddings = HuggingFaceEmbeddings(model_name=self.config.model_name)  # 修正

        # カスタムエンベッディング関数の作成
        embedding_function = lambda x: self.encode_query(x)

        self.vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            embedding=embedding_function,
            metadatas=[chunk["metadata"] for chunk in all_chunks],
        )

        # 保存
        save_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(save_dir / "vectorstore"))

        with open(save_dir / "document_map.pkl", "wb") as f:
            pickle.dump(self.document_map, f)

        # インデックス情報の保存
        index_info = {
            "total_chunks": len(all_chunks),
            "industries": [d.name for d in industry_dirs],
            "processed_files": [
                str(pdf_path.relative_to(base_dir))
                for industry_dir in industry_dirs
                for pdf_path in industry_dir.glob("*.pdf")
            ],
            "timestamp": datetime.now().isoformat(),
        }

        with open(save_dir / "index_info.json", "w", encoding="utf-8") as f:
            json.dump(index_info, f, ensure_ascii=False, indent=2)

        logging.info(f"Index built successfully. Total chunks: {len(all_chunks)}")

    def load_index(self, save_dir: Path):
        """保存されたインデックスの読み込み"""
        if not (save_dir / "vectorstore").exists():
            raise ValueError(f"No index found at {save_dir}")

        # カスタムエンベッディング関数の作成
        embedding_function = lambda x: self.encode_query(x)

        self.vectorstore = FAISS.load_local(
            str(save_dir / "vectorstore"),
            embedding_function,
            allow_dangerous_deserialization=True,
        )

        with open(save_dir / "document_map.pkl", "rb") as f:
            self.document_map = pickle.load(f)

    def search(self, query: str, filter_criteria: Optional[Dict] = None) -> List[Dict]:
        """検索の実行"""
        if self.vectorstore is None:
            raise ValueError("Index not built or loaded")

        # 検索の実行前にクエリを前処理
        query = self.processor.preprocessor.clean_text(query)

        # 検索の実行
        results = self.vectorstore.similarity_search_with_score(
            query, k=self.config.max_results
        )

        # 結果の整形とフィルタリング
        processed_results = []
        for doc, score in results:
            if score >= self.config.similarity_threshold:
                metadata = doc.metadata

                # フィルタリング条件がある場合は適用
                if filter_criteria:
                    if not self._matches_criteria(metadata, filter_criteria):
                        continue

                processed_results.append(
                    {
                        "content": doc.page_content,
                        "metadata": metadata,
                        "score": float(score),
                    }
                )

        # 検索結果のロギングを改善 (追加)
        for result in processed_results:
            logging.info(
                f" - Score: {result['score']:.4f}, Content: {result['content'][:50]}..."
            )

        return processed_results

    def _batch_encode(self, texts: List[str]) -> np.ndarray:
        """テキストのバッチエンコード"""
        embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            with torch.amp.autocast(device_type=self.config.device):
                batch_embeddings = self.model.encode(
                    batch, convert_to_tensor=True, show_progress_bar=False
                )
            embeddings.extend(batch_embeddings.cpu().numpy())
        return np.array(embeddings)

    @staticmethod
    def _matches_criteria(metadata: Dict, criteria: Dict) -> bool:
        """メタデータがフィルタリング条件に一致するかチェック"""
        return all(metadata.get(key) == value for key, value in criteria.items())


def main():
    # ログの設定
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # パスの設定
    base_dir = Path("/root_nas05/home/2022/naoki/AI-Scientist/data/esg")
    raw_dir = base_dir / "raw"
    index_dir = base_dir / "processed" / "search_index"

    # 設定
    config = SearchConfig(
        chunk_size=300, chunk_overlap=50, batch_size=16, similarity_threshold=0.2
    )

    try:
        # 検索システムの初期化
        search_system = ESGSearchSystem(config)

        # インデックスの構築（または読み込み）
        if not index_dir.exists() or not (index_dir / "vectorstore").exists():
            logging.info("Building new search index...")
            search_system.build_index(raw_dir, index_dir)
        else:
            logging.info("Loading existing search index...")
            search_system.load_index(index_dir)

        # サンプルクエリでのテスト
        sample_queries = [
            "CO2排出量の削減目標と実績について",
            "従業員の安全衛生に関する取り組み",
            "ガバナンス体制の整備状況",
        ]

        # 業界ごとの結果を取得
        results = {}
        for industry in ["自動車", "情報通信", "小売"]:
            industry_results = {}
            for query in sample_queries:
                logging.info(f"\nExecuting search for {industry} - {query}")
                filter_criteria = {"industry": industry}
                search_results = search_system.search(query, filter_criteria)
                industry_results[query] = search_results
                logging.info(f"Found {len(search_results)} relevant chunks")
            results[industry] = industry_results

        # 結果の保存
        output_dir = base_dir / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(
            output_dir / f"search_results_{timestamp}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logging.info("\nSearch system test completed successfully")

    except Exception as e:
        logging.error(f"Error during execution: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
