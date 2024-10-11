import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    pipeline,
)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
from langchain_community.vectorstores import FAISS
import json
import time

# Intel MKLを使用するための環境変数を設定
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# numpyがIntel MKLを使用していることを確認
np.__config__.show()

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
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    vectorstore = FAISS.load_local(
        os.path.join(data_dir, "vectorstore"),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta_data = pickle.load(f)
    return vectorstore, meta_data


def setup_swallow_model():
    model_name = "tokyotech-llm/Swallow-MX-8x7b-NVE-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    def generate_text(prompt, max_new_tokens=128):
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.99,
            top_p=0.95,
            do_sample=True,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generate_text


def setup_rag_chain(vectorstore, qa_model, generate_text):
    retriever = vectorstore.as_retriever()

    def rag_chain(query):
        docs = retriever.get_relevant_documents(query)
        context = " ".join([doc.page_content for doc in docs])

        qa_input = f"コンテキスト: {context}\n質問: {query}"
        qa_result = qa_model(question=query, context=context)

        summary_prompt = f"以下の文章を要約してください：\n{qa_result['answer']}"
        summary = generate_text(summary_prompt)

        insight_prompt = (
            f"以下の情報に基づいて、投資判断に役立つ洞察を生成してください：\n{summary}"
        )
        insights = generate_text(insight_prompt)

        return f"回答: {qa_result['answer']}\n\n要約: {summary}\n\n投資洞察: {insights}"

    return rag_chain


def analyze_and_generate_new_question(
    response, initial_question, esg_topics, generate_text
):
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    response_embedding = model.encode(response)
    topic_embeddings = model.encode(esg_topics)

    similarities = cosine_similarity([response_embedding], topic_embeddings)[0]

    low_similarity_topics = [esg_topics[i] for i in np.argsort(similarities)[:3]]

    new_question_prompt = f"初期質問「{initial_question}」と回答に基づいて、{', '.join(low_similarity_topics)}に関する追加の質問を生成してください。"
    new_question = generate_text(new_question_prompt, max_new_tokens=100)

    return new_question, similarities


def iterative_extraction(rag_chain, initial_question, generate_text, max_iterations=3):
    current_question = initial_question
    all_responses = []
    iteration_quality = []

    esg_topics = [topic for category in esg_categories.values() for topic in category]

    for i in range(max_iterations):
        print(f"\nイテレーション {i+1}/{max_iterations} 開始")
        start_time = time.time()

        print("  RAGチェーンを実行中...")
        response = rag_chain(current_question)
        all_responses.append(response)

        print("  新しい質問を生成中...")
        new_question, similarities = analyze_and_generate_new_question(
            response, initial_question, esg_topics, generate_text
        )
        current_question = new_question

        iteration_quality.append(np.mean(similarities))

        end_time = time.time()
        print(f"イテレーション {i+1} 完了 (所要時間: {end_time - start_time:.2f}秒)")

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


def main(output_dir):
    print("ESG分析を開始します...")
    start_time = time.time()

    data_dir = "/root_nas05/home/2022/naoki/AI-Scientist/data/esg"
    os.makedirs(output_dir, exist_ok=True)

    print("準備済みデータを読み込んでいます...")
    vectorstore, meta_data = load_prepared_data(data_dir)

    print("質問応答モデルを設定しています...")
    qa_model = pipeline(
        "question-answering",
        model="tsmatz/roberta_qa_japanese",
    )

    print("Swallowモデルを設定しています...")
    generate_text = setup_swallow_model()

    print("RAGチェーンを設定しています...")
    rag_chain = setup_rag_chain(vectorstore, qa_model, generate_text)

    initial_question = "このレポートで議論されている主要なESG要因は何で、それらが投資判断にどのような影響を与える可能性がありますか？"
    print("反復抽出を開始します...")
    final_response, iteration_quality = iterative_extraction(
        rag_chain, initial_question, generate_text
    )

    print("結果をファイルに書き込んでいます...")
    with open(
        os.path.join(output_dir, "esg_analysis_result.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("ESG分析結果:\n\n")
        f.write(f"初期質問: {initial_question}\n\n")
        f.write("反復抽出の結果:\n")
        f.write(final_response)

    print("抽出結果を評価しています...")
    topic_coverage = evaluate_extraction(
        final_response,
        [topic for category in esg_categories.values() for topic in category],
    )

    results = {
        "topic_coverage": dict(
            zip(
                [topic for category in esg_categories.values() for topic in category],
                [float(x) for x in topic_coverage],  # Convert float32 to float
            )
        ),
        "iteration_quality": [
            float(x) for x in iteration_quality
        ],  # Convert float32 to float
    }

    print("評価結果をJSONファイルに書き込んでいます...")
    with open(
        os.path.join(output_dir, "esg_analysis_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 評価結果をより明確なテキストファイルとしても出力
    with open(
        os.path.join(output_dir, "esg_analysis_summary.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("ESG分析評価結果サマリー:\n\n")
        f.write("トピックカバレッジ:\n")
        for topic, coverage in results["topic_coverage"].items():
            f.write(f"  {topic}: {coverage:.4f}\n")
        f.write("\n反復品質:\n")
        for i, quality in enumerate(results["iteration_quality"]):
            f.write(f"  イテレーション {i+1}: {quality:.4f}\n")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nESG分析が完了しました。総実行時間: {total_time:.2f}秒")

    # 実行時間も結果ファイルに追加
    with open(
        os.path.join(output_dir, "esg_analysis_summary.txt"), "a", encoding="utf-8"
    ) as f:
        f.write(f"\n総実行時間: {total_time:.2f}秒\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ESG analysis")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()

    main(args.out_dir)
