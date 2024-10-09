import os
from tqdm import tqdm
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import functools
from ja_sentence_segmenter.common.pipeline import make_pipeline
from ja_sentence_segmenter.concatenate.simple_concatenator import concatenate_matching
from ja_sentence_segmenter.normalize.neologd_normalizer import normalize
from ja_sentence_segmenter.split.simple_splitter import split_newline, split_punctuation

# 日本語文書の分割器の設定
split_punc2 = functools.partial(split_punctuation, punctuations=r"。!?")
concat_tail_te = functools.partial(
    concatenate_matching,
    former_matching_rule=r"^(?P<result>.+)(て)$",
    remove_former_matched=False,
)
segmenter = make_pipeline(normalize, split_newline, concat_tail_te, split_punc2)


def load_and_process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 日本語文書の分割
    segmented_documents = []
    for doc in documents:
        segmented_text = list(segmenter(doc.page_content))
        segmented_documents.append("\n".join(segmented_text))

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(segmented_documents)

    return texts


def create_vectorstore(texts):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore


def main():
    input_dir = "/root_nas05/home/2022/naoki/AI-Scientist/data/esg"
    output_dir = "/root_nas05/home/2022/naoki/AI-Scientist/data/esg"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_texts = []
    for pdf_file in tqdm(os.listdir(input_dir)):
        if pdf_file.endswith(".pdf"):
            file_path = os.path.join(input_dir, pdf_file)
            texts = load_and_process_pdf(file_path)
            all_texts.extend(texts)

    print(f"総チャンク数: {len(all_texts)}")

    # ベクトルストアの作成
    vectorstore = create_vectorstore(all_texts)

    # ベクトルストアの保存
    vectorstore.save_local(os.path.join(output_dir, "vectorstore"))

    # メタデータの保存
    meta_data = {
        "total_chunks": len(all_texts),
        "embedding_model": "paraphrase-multilingual-mpnet-base-v2",
    }
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta_data, f)


if __name__ == "__main__":
    main()
