import argparse
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
from langchain.vectorstores import FAISS
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
    # ベクトルストアの読み込み
    vectorstore = FAISS.load_local(os.path.join(data_dir, "vectorstore"))

    # メタデータの読み込み
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta_data = pickle.load(f)

    return vectorstore, meta_data


def setup_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    return qa_chain


def analyze_and_generate_new_question(response, initial_question, esg_topics):
    # SentenceTransformerを使用してより高度な類似度計算を行う
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    response_embedding = model.encode(response)
    topic_embeddings = model.encode(esg_topics)

    similarities = cosine_similarity([response_embedding], topic_embeddings)[0]

    # 類似度が低いトピックを特定
    low_similarity_topics = [esg_topics[i] for i in np.argsort(similarities)[:3]]

    # 新しい質問を生成（日本語のDeBERTaモデルを使用）
    question_generator = pipeline(
        "text-generation", model="izumi-lab/deberta-v2-base-japanese"
    )
    new_question_prompt = f"初期質問「{initial_question}」と回答に基づいて、{', '.join(low_similarity_topics)}に関する追加の質問を生成してください。"
    new_question = question_generator(
        new_question_prompt, max_length=100, num_return_sequences=1
    )[0]["generated_text"]

    return new_question, similarities


def iterative_extraction(qa_chain, initial_question, max_iterations=3):
    current_question = initial_question
    all_responses = []
    iteration_quality = []

    esg_topics = [topic for category in esg_categories.values() for topic in category]

    for i in range(max_iterations):
        response = qa_chain.run(current_question)
        all_responses.append(response)

        new_question, similarities = analyze_and_generate_new_question(
            response, initial_question, esg_topics
        )
        current_question = new_question

        # イテレーションの品質を計算（ここでは単純に平均類似度を使用）
        iteration_quality.append(np.mean(similarities))

    final_response = "\n\n".join(all_responses)
    return final_response, iteration_quality


def main():
    data_dir = "/root_nas05/home/2022/naoki/AI-Scientist/data/esg"
    output_dir = "/root_nas05/home/2022/naoki/AI-Scientist/templates/esg"

    vectorstore, meta_data = load_prepared_data(data_dir)

    # 日本語のDeBERTaモデルを使用
    tokenizer = AutoTokenizer.from_pretrained("izumi-lab/deberta-v2-base-japanese")
    model = AutoModel.from_pretrained("izumi-lab/deberta-v2-base-japanese")
    llm = HuggingFacePipeline.from_model_id(
        model_id="izumi-lab/deberta-v2-base-japanese",
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    qa_chain = setup_rag_chain(vectorstore, llm)

    initial_question = "このレポートで議論されている主要なESG要因は何で、それらが投資判断にどのような影響を与える可能性がありますか？"
    final_response, iteration_quality = iterative_extraction(qa_chain, initial_question)

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


if __name__ == "__main__":
    main()
