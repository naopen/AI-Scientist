from pathlib import Path
import torch
from tqdm import tqdm
import pickle
from typing import List
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


class SplitMode(Enum):
    CHUNK = "chunk"
    SENTENCE = "sentence"


@dataclass
class ProcessingConfig:
    # 分割モード設定
    split_mode: SplitMode = SplitMode.CHUNK
    # チャンク分割用の設定
    chunk_size: int = 500
    chunk_overlap: int = 50
    # 文単位分割用の設定
    max_sentences: int = 3
    sentence_overlap: int = 1
    # 例示用の設定
    num_examples: int = 5  # 表示する例の数
    start_index: int = 0  # 表示を開始する位置（0から開始）
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CustomEmbeddings(HuggingFaceEmbeddings):
    def __init__(self):
        super().__init__(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={"device": ProcessingConfig.device},
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
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.embeddings = CustomEmbeddings()

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
        """分割された文書の例を表示"""
        total_docs = len(documents)
        start_idx = min(self.config.start_index, total_docs)
        end_idx = min(start_idx + self.config.num_examples, total_docs)

        print(f"\n=== 分割例 ({file_name}) ===")
        print(f"分割モード: {self.config.split_mode.value}")
        print(f"総文書数: {total_docs}")
        print(f"表示範囲: {start_idx + 1}番目 ～ {end_idx}番目")

        if self.config.split_mode == SplitMode.CHUNK:
            print(f"チャンクサイズ: {self.config.chunk_size}")
            print(f"オーバーラップ: {self.config.chunk_overlap}")
        else:
            print(f"文数: {self.config.max_sentences}")
            print(f"文オーバーラップ: {self.config.sentence_overlap}")

        print("\n=== 分割結果 ===")
        for i, doc in enumerate(documents[start_idx:end_idx], start=start_idx + 1):
            print(f"\n例 {i}:")
            print(f"長さ: {len(doc.page_content)} 文字")
            print("内容:")  # 別行で内容を表示
            print(doc.page_content)  # 内容を別行で表示
            print("-" * 80)

    def process_pdf(self, file_path: Path) -> List[Document]:
        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()

            if self.config.split_mode == SplitMode.CHUNK:
                text_splitter = CharacterTextSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )
                docs = text_splitter.split_documents(documents)
            else:
                docs = []
                for doc in documents:
                    sentences = self.split_into_sentences(doc.page_content)
                    windows = self.create_sentence_windows(sentences)

                    for window in windows:
                        docs.append(
                            Document(
                                page_content=window,
                                metadata={
                                    **doc.metadata,
                                    "split_type": "sentence",
                                    "window_size": self.config.max_sentences,
                                },
                            )
                        )

            # 分割例を表示
            self.display_examples(docs, file_path.name)
            return docs

        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            return []

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        return FAISS.from_documents(documents, self.embeddings)

    def process_directory(self, input_dir: Path, output_dir: Path):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_documents = []
        pdf_files = list(input_dir.glob("*.pdf"))

        if not pdf_files:
            logging.error(f"No PDF files found in {input_dir}")
            return

        for pdf_file in tqdm(pdf_files):
            try:
                docs = self.process_pdf(pdf_file)
                if docs:
                    all_documents.extend(docs)
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
            vector_store = self.create_vector_store(all_documents)
            vector_store.save_local(str(output_dir / "vectorstore"))

            meta_data = {
                "total_chunks": len(all_documents),
                "embedding_model": "intfloat/multilingual-e5-large",
                "split_mode": self.config.split_mode.value,
            }
            with open(output_dir / "metadata.pkl", "wb") as f:
                pickle.dump(meta_data, f)

            logging.info("Successfully created and saved vector store")
        except Exception as e:
            logging.error(f"Error creating vector store: {str(e)}")


def main():
    """
    実行モードの設定
    TEST_MODE: True = テスト実行（サンプル表示あり）, False = 本番実行（サンプル表示なし）
    """
    TEST_MODE = True  # ここを False に切り替えることで本番実行モードになります

    # ============= テスト実行用の設定 =============
    if TEST_MODE:
        test_configs = [
            # テスト用：チャンク分割（少量のサンプルを表示）
            ProcessingConfig(
                split_mode=SplitMode.CHUNK,
                chunk_size=500,
                chunk_overlap=50,
                num_examples=5,
                start_index=50,
            ),
            # テスト用：文単位分割（少量のサンプルを表示）
            ProcessingConfig(
                split_mode=SplitMode.SENTENCE,
                max_sentences=3,
                sentence_overlap=1,
                num_examples=5,
                start_index=50,
            ),
        ]

        for config in test_configs:
            print(f"\n=== テスト実行: {config.split_mode.value} モード ===")
            processor = ESGDocumentProcessor(config)
            input_dir = Path("/root_nas05/home/2022/naoki/AI-Scientist/data/esg")
            output_dir = Path(
                "/root_nas05/home/2022/naoki/AI-Scientist/data/esg/processed_test"
            )
            processor.process_directory(input_dir, output_dir)

    # ============= 本番実行用の設定 =============
    else:
        config = ProcessingConfig(
            split_mode=SplitMode.CHUNK,  # チャンク分割を使用する場合
            # split_mode=SplitMode.SENTENCE, # 文単位分割を使用する場合
            max_sentences=3,
            sentence_overlap=1,
            num_examples=0,  # 本番環境では例を表示しない
            start_index=0,
        )

        processor = ESGDocumentProcessor(config)
        input_dir = Path("/root_nas05/home/2022/naoki/AI-Scientist/data/esg")
        output_dir = Path("/root_nas05/home/2022/naoki/AI-Scientist/data/esg/processed")
        processor.process_directory(input_dir, output_dir)


if __name__ == "__main__":
    main()
