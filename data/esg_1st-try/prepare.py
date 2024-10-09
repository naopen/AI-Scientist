"""
prepare.py - 日本語RAGシステムのためのデータ準備スクリプト

このスクリプトは以下の機能を持ちます：
1. 日本語PDFファイルからのテキスト抽出 (PyPDFLoaderを使用)
2. 高度な日本語テキストの文分割と前処理
3. 文レベルのエンコーディング
4. enwik8形式でのデータ保存（train.bin, val.bin, test.bin, meta.pkl）
"""

import os
import pickle
import numpy as np

# PyPDF2の代わりにPyPDFLoaderを使用
from langchain_community.document_loaders import PyPDFLoader
import MeCab
import re
import functools
from collections import defaultdict

from ja_sentence_segmenter.common.pipeline import make_pipeline
from ja_sentence_segmenter.concatenate.simple_concatenator import concatenate_matching
from ja_sentence_segmenter.normalize.neologd_normalizer import normalize
from ja_sentence_segmenter.split.simple_splitter import split_newline, split_punctuation

# 文分割の設定
split_punc2 = functools.partial(split_punctuation, punctuations=r"。!?")
concat_tail_te = functools.partial(
    concatenate_matching,
    former_matching_rule=r"^(?P<result>.+)(て)$",
    remove_former_matched=False,
)
segmenter = make_pipeline(normalize, split_newline, concat_tail_te, split_punc2)


def extract_text_from_pdf(pdf_path):
    """PDFファイルからテキストを抽出する (PyPDFLoaderを使用)"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text = ""
    for doc in documents:
        text += doc.page_content + "\n"
    print("Extracted text:")  # デバッグ出力
    print(text)
    return text


def preprocess_sentence(sentence):
    """日本語の文の基本的な前処理を行う"""
    # MeCabの初期化（Condaでインストールした場合）
    mecab = MeCab.Tagger("-Owakati")

    # 不要な文字の削除（スペースのみ）
    sentence = re.sub(r"[\s　]", "", sentence)

    # MeCabを使用して分かち書きを行う
    tokens = mecab.parse(sentence).strip().split()
    print("Tokens:")  # デバッグ出力
    print(tokens)

    return " ".join(tokens)


def create_sentence_encoding(sentences):
    """文レベルのエンコーディングを作成"""
    unique_sentences = list(set(sentences))
    sentence_to_idx = {sent: i for i, sent in enumerate(unique_sentences)}
    idx_to_sentence = {i: sent for i, sent in enumerate(unique_sentences)}
    return unique_sentences, sentence_to_idx, idx_to_sentence


def encode_sentences(sentences, sentence_to_idx):
    """文をエンコード"""
    return [sentence_to_idx[sent] for sent in sentences]


# PDFファイルのディレクトリとデータの出力先を指定
pdf_directory = "."
output_directory = "."

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

all_sentences = []

# PDFファイルのリストを出力
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]
print("PDF files:")
print(pdf_files)


# PDFファイルからテキストを抽出し、文分割と前処理を行う
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        print(f"Processing {filename}...")

        try:
            text = extract_text_from_pdf(pdf_path)
            sentences = list(segmenter(text))
            print("Sentences:")  # デバッグ出力
            print(sentences)
            processed_sentences = [preprocess_sentence(sent) for sent in sentences]
            all_sentences.extend(processed_sentences)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

print("Creating sentence-level encoding...")
unique_sentences, sentence_to_idx, idx_to_sentence = create_sentence_encoding(
    all_sentences
)
encoded_data = encode_sentences(all_sentences, sentence_to_idx)

# データの分割
n = len(encoded_data)
train_data = encoded_data[: int(0.8 * n)]
val_data = encoded_data[int(0.8 * n) : int(0.9 * n)]
test_data = encoded_data[int(0.9 * n) :]

print("Saving data...")
# train.bin, val.bin, test.bin の保存
np.array(train_data, dtype=np.uint32).tofile(
    os.path.join(output_directory, "train.bin")
)
np.array(val_data, dtype=np.uint32).tofile(os.path.join(output_directory, "val.bin"))
test_data = np.array(test_data, dtype=np.uint32)
test_data.tofile(os.path.join(output_directory, "test.bin"))

# meta.pkl の保存
meta = {
    "vocab_size": len(unique_sentences),
    "itos": idx_to_sentence,
    "stoi": sentence_to_idx,
}
with open(os.path.join(output_directory, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("Data preparation completed.")
print(f"Unique sentences: {len(unique_sentences)}")
print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(val_data)}")
print(f"Test size: {len(test_data)}")
