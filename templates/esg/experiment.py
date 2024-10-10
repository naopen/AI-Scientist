import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    pipeline,
)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.llms import HuggingFacePipeline
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
from langchain_community.vectorstores import FAISS
import json

# ESGカテゴリとサブカテゴリの定義
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


def load_prepared_data(data_dir):
    # エンベディングモデルのインスタンスを作成
    embeddings = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    # ベクトルストアの読み込み
    vectorstore = FAISS.load_local(
        os.path.join(data_dir, "vectorstore"),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    # メタデータの読み込み
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta_data = pickle.load(f)

    return vectorstore, meta_data


def setup_rag_chain(vectorstore, qa_model, summarizer, generator):
    retriever = vectorstore.as_retriever()

    def rag_chain(query):
        # 関連文書の取得
        docs = retriever.get_relevant_documents(query)
        context = " ".join([doc.page_content for doc in docs])

        # 質問応答
        qa_input = f"コンテキスト: {context}\n質問: {query}"
        qa_result = qa_model(qa_input)

        # 要約
        summary = summarizer(
            qa_result["answer"], max_length=150, min_length=50, do_sample=False
        )[0]["summary_text"]

        # 投資判断に関連する洞察の生成
        prompt = (
            f"以下の情報に基づいて、投資判断に役立つ洞察を生成してください：\n{summary}"
        )
        insights = generator(prompt, max_length=200, num_return_sequences=1)[0][
            "generated_text"
        ]

        return f"回答: {qa_result['answer']}\n\n要約: {summary}\n\n投資洞察: {insights}"

    return rag_chain


def analyze_and_generate_new_question(
    response, initial_question, esg_topics, generator
):
    # SentenceTransformerを使用してより高度な類似度計算を行う
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    response_embedding = model.encode(response)
    topic_embeddings = model.encode(esg_topics)

    similarities = cosine_similarity([response_embedding], topic_embeddings)[0]

    # 類似度が低いトピックを特定
    low_similarity_topics = [esg_topics[i] for i in np.argsort(similarities)[:3]]

    # 新しい質問を生成
    new_question_prompt = f"初期質問「{initial_question}」と回答に基づいて、{', '.join(low_similarity_topics)}に関する追加の質問を生成してください。"
    new_question = generator(
        new_question_prompt, max_length=100, num_return_sequences=1
    )[0]["generated_text"]

    return new_question, similarities


def iterative_extraction(rag_chain, initial_question, generator, max_iterations=3):
    current_question = initial_question
    all_responses = []
    iteration_quality = []

    esg_topics = [topic for category in esg_categories.values() for topic in category]

    for i in range(max_iterations):
        response = rag_chain(current_question)
        all_responses.append(response)

        new_question, similarities = analyze_and_generate_new_question(
            response, initial_question, esg_topics, generator
        )
        current_question = new_question

        # イテレーションの品質を計算（ここでは単純に平均類似度を使用）
        iteration_quality.append(np.mean(similarities))

    final_response = "\n\n".join(all_responses)
    return final_response, iteration_quality


def evaluate_extraction(final_response, esg_topics):
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    response_embedding = model.encode(final_response)
    topic_embeddings = model.encode(esg_topics)

    similarities = cosine_similarity([response_embedding], topic_embeddings)[0]

    print("ESGトピックのカバレッジ:")
    for topic, similarity in zip(esg_topics, similarities):
        print(f"{topic}: {similarity:.2f}")

    return similarities.tolist()


def main():
    data_dir = "/root_nas05/home/2022/naoki/AI-Scientist/data/esg"
    output_dir = "/root_nas05/home/2022/naoki/AI-Scientist/templates/esg"

    vectorstore, meta_data = load_prepared_data(data_dir)

    # モデルの設定
    qa_model = pipeline(
        "question-answering",
        model="cl-tohoku/bert-base-japanese-whole-word-masking-squad",
    )
    summarizer = pipeline(
        "summarization", model="ku-nlp/deberta-v2-base-japanese-squad"
    )
    generator = pipeline("text-generation", model="rinna/japanese-gpt-neox-3.6b")

    rag_chain = setup_rag_chain(vectorstore, qa_model, summarizer, generator)

    initial_question = "このレポートで議論されている主要なESG要因は何で、それらが投資判断にどのような影響を与える可能性がありますか？"
    final_response, iteration_quality = iterative_extraction(
        rag_chain, initial_question, generator
    )

    with open(
        os.path.join(output_dir, "esg_analysis_result.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(final_response)

    topic_coverage = evaluate_extraction(
        final_response,
        [topic for category in esg_categories.values() for topic in category],
    )

    # 結果をJSONファイルに保存
    results = {
        "topic_coverage": dict(
            zip(
                [topic for category in esg_categories.values() for topic in category],
                topic_coverage,
            )
        ),
        "iteration_quality": iteration_quality,
    }
    with open(
        os.path.join(output_dir, "esg_analysis_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
