import argparse
import torch
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
from datetime import datetime

# Intel MKLを使用するための環境変数を設定
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# numpyがIntel MKLを使用していることを確認
# np.__config__.show()

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

# 詳細な参照データの定義
reference_esg_data = {
    "Environmental": {
        "気候変動対策": {
            "content": "企業はCO2排出量の削減目標を設定し、再生可能エネルギーの導入を進めている。また、カーボンプライシングの導入も検討している。",
            "facts": [
                "CO2排出量削減目標",
                "再生可能エネルギー導入",
                "カーボンプライシング検討",
            ],
            "impact": "長期的な環境規制リスクの低減と運用コストの削減につながる可能性がある。また、炭素税導入時のリスク軽減にも寄与する。",
        },
        "再生可能エネルギー": {
            "content": "太陽光発電や風力発電などの再生可能エネルギー施設への投資を増やし、エネルギー調達の多様化を図っている。",
            "facts": ["太陽光発電投資", "風力発電投資", "エネルギー調達多様化"],
            "impact": "長期的なエネルギーコストの安定化と、環境配慮型企業としてのブランド価値向上が期待できる。",
        },
        "廃棄物管理": {
            "content": "製品のリサイクル率向上と廃棄物削減のための取り組みを強化している。また、サーキュラーエコノミーの概念を導入している。",
            "facts": ["リサイクル率向上", "廃棄物削減", "サーキュラーエコノミー導入"],
            "impact": "資源効率の向上によるコスト削減と、環境負荷低減による企業イメージの向上が見込まれる。",
        },
        "水資源管理": {
            "content": "水使用効率の改善と水質汚濁防止のための設備投資を行っている。また、水ストレス地域での事業における水リスク評価を実施している。",
            "facts": ["水使用効率改善", "水質汚濁防止投資", "水リスク評価"],
            "impact": "水関連コストの削減と、水資源をめぐる地域社会との潜在的な紛争リスクの低減が期待できる。",
        },
        "エネルギー効率": {
            "content": "省エネ技術の導入と、エネルギーマネジメントシステムの構築を進めている。また、建物や設備の省エネ改修も実施している。",
            "facts": [
                "省エネ技術導入",
                "エネルギーマネジメントシステム構築",
                "省エネ改修",
            ],
            "impact": "エネルギーコストの削減と、環境規制への適応力向上によるビジネスリスクの低減が見込まれる。",
        },
        "生物多様性": {
            "content": "事業活動が生態系に与える影響を評価し、生物多様性の保全活動を実施している。また、サプライチェーン全体での生物多様性への配慮も促進している。",
            "facts": [
                "生態系影響評価",
                "生物多様性保全活動",
                "サプライチェーンでの配慮",
            ],
            "impact": "生態系サービスへの依存リスクの軽減と、生物多様性に配慮した企業としてのレピュテーション向上が期待できる。",
        },
    },
    "Social": {
        "労働条件": {
            "content": "従業員の労働環境改善と適切な報酬制度の導入に取り組んでいる。また、ワークライフバランスの推進と柔軟な勤務形態の導入を行っている。",
            "facts": [
                "労働環境改善",
                "適切な報酬制度",
                "ワークライフバランス推進",
                "柔軟な勤務形態",
            ],
            "impact": "従業員の生産性向上とリテンション率の改善により、長期的な企業価値向上が期待できる。また、優秀な人材の獲得にも寄与する。",
        },
        "人権": {
            "content": "人権デューデリジェンスを実施し、サプライチェーン全体での人権尊重を推進している。また、人権に関する教育・研修プログラムを従業員に提供している。",
            "facts": [
                "人権デューデリジェンス",
                "サプライチェーンでの人権尊重",
                "人権教育プログラム",
            ],
            "impact": "人権侵害に関するリスクの低減と、社会的責任を果たす企業としての評価向上が見込まれる。",
        },
        "ダイバーシティ": {
            "content": "性別、年齢、国籍、障害の有無などに関わらず、多様な人材の登用と活躍推進を行っている。また、インクルーシブな職場環境の構築に取り組んでいる。",
            "facts": [
                "多様な人材登用",
                "インクルーシブな職場環境",
                "ダイバーシティ推進施策",
            ],
            "impact": "イノベーション創出力の向上と、多様な市場ニーズへの対応力強化が期待できる。",
        },
        "従業員の健康と安全": {
            "content": "労働安全衛生マネジメントシステムの導入と、メンタルヘルスケアを含む健康管理プログラムの実施を行っている。また、新型コロナウイルス対策も強化している。",
            "facts": [
                "労働安全衛生マネジメント",
                "健康管理プログラム",
                "メンタルヘルスケア",
                "感染症対策",
            ],
            "impact": "労働災害リスクの低減と、従業員の健康増進による生産性向上が見込まれる。また、パンデミック等の健康危機への耐性強化にも寄与する。",
        },
        "サプライチェーン管理": {
            "content": "サプライヤーの労働条件や環境対応をモニタリングし、持続可能な調達方針を策定している。また、サプライヤーの能力開発支援も行っている。",
            "facts": [
                "サプライヤーモニタリング",
                "持続可能な調達方針",
                "サプライヤー能力開発",
            ],
            "impact": "サプライチェーンリスクの低減と、安定的な調達体制の構築による事業継続性の向上が期待できる。",
        },
        "地域社会への貢献": {
            "content": "事業活動を通じた地域経済への貢献と、社会貢献活動の実施を行っている。また、地域コミュニティとの対話を通じて、地域のニーズに応じた取り組みを推進している。",
            "facts": ["地域経済貢献", "社会貢献活動", "地域コミュニティとの対話"],
            "impact": "地域社会との良好な関係構築による事業環境の安定化と、企業ブランド価値の向上が見込まれる。",
        },
    },
    "Governance": {
        "コーポレートガバナンス": {
            "content": "取締役会の多様性を高め、独立した監査委員会を設置している。また、経営の透明性向上のための情報開示の拡充を行っている。",
            "facts": ["取締役会の多様性", "独立監査委員会", "情報開示の拡充"],
            "impact": "経営の健全性と透明性の向上により、投資家の信頼度が向上する可能性がある。また、不正リスクの低減にも寄与する。",
        },
        "リスク管理": {
            "content": "統合的リスク管理システムを構築し、定期的なリスク評価と対応策の策定を行っている。また、気候変動関連リスクの評価とTCFD提言に基づく情報開示も実施している。",
            "facts": [
                "統合的リスク管理",
                "定期的リスク評価",
                "気候関連リスク評価",
                "TCFD対応",
            ],
            "impact": "事業継続性の向上と、投資家に対する適切なリスク情報の提供による信頼性向上が期待できる。",
        },
        "コンプライアンス": {
            "content": "行動規範の策定と従業員への浸透、内部通報制度の整備を行っている。また、定期的なコンプライアンス研修の実施と、違反事例の分析・対策も行っている。",
            "facts": [
                "行動規範策定",
                "内部通報制度",
                "コンプライアンス研修",
                "違反事例分析",
            ],
            "impact": "法令違反リスクの低減と、企業倫理の向上による企業価値の保護が見込まれる。",
        },
        "取締役会の構成": {
            "content": "社外取締役の比率を高め、取締役会の実効性評価を定期的に実施している。また、取締役のスキルマトリックスの開示も行っている。",
            "facts": [
                "社外取締役比率向上",
                "取締役会実効性評価",
                "スキルマトリックス開示",
            ],
            "impact": "経営の客観性と透明性の向上により、投資家からの信頼獲得と適切な経営判断の促進が期待できる。",
        },
        "株主の権利": {
            "content": "株主総会における議決権行使の利便性向上と、株主との建設的な対話の促進を図っている。また、少数株主の権利保護にも留意している。",
            "facts": ["議決権行使利便性向上", "株主との対話促進", "少数株主権利保護"],
            "impact": "株主からの信頼獲得と、長期的な企業価値向上に向けた株主との協力関係構築が見込まれる。",
        },
        "経営の透明性": {
            "content": "統合報告書の発行や、ESG情報の積極的な開示を行っている。また、経営戦略や中長期的な価値創造プロセスの明確な説明にも注力している。",
            "facts": ["統合報告書発行", "ESG情報開示", "価値創造プロセス説明"],
            "impact": "投資家の理解促進と、企業価値の適切な評価につながる可能性がある。また、ステークホルダーとの信頼関係構築にも寄与する。",
        },
    },
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

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        temperature=0.99,
        top_p=0.95,
        do_sample=True,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return llm


def setup_improved_rag_chain(vectorstore, llm):
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm_chain_extractor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=llm_chain_extractor, base_retriever=base_retriever
    )

    def rag_chain(query):
        relevant_documents = compression_retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in relevant_documents])

        prompt = f"""
        以下のコンテキストに基づいて、質問に答えてください。
        できるだけ詳細かつ具体的に回答し、ESGの観点から投資判断に役立つ情報を提供してください。

        コンテキスト:
        {context}

        質問: {query}

        回答:
        """

        response = llm(prompt)

        return response, context

    return rag_chain


def generate_focused_question(context, esg_categories, llm):
    prompt = f"""
    以下の文脈に基づいて、ESGと投資関連性に焦点を当てた質問を生成してください。
    文脈: {context}
    
    ESGカテゴリ:
    {json.dumps(esg_categories, ensure_ascii=False, indent=2)}
    
    生成する質問は以下の条件を満たすようにしてください:
    1. ESGの特定のカテゴリまたはサブカテゴリに関連していること
    2. 投資判断に直接影響を与える可能性のある情報を求めていること
    3. 具体的で明確であること
    4. 文脈から得られた情報を深掘りするものであること
    
    質問:
    """
    return llm(prompt)


def extract_structured_info(text, esg_categories, llm):
    structured_info = {category: {} for category in esg_categories}

    for category, subcategories in esg_categories.items():
        for subcategory in subcategories:
            prompt = f"""
            以下の文章から、{category}カテゴリの{subcategory}に関する情報を抽出し、
            簡潔にまとめてください。投資判断に関連する情報に特に注目してください。

            文章:
            {text}

            抽出した情報（100字以内）:
            """
            extracted_info = llm(prompt)
            structured_info[category][subcategory] = extracted_info

    return structured_info


def analyze_and_generate_new_question(
    response, context, initial_question, esg_topics, llm
):
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    response_embedding = model.encode(response)
    topic_embeddings = model.encode(esg_topics)

    similarities = cosine_similarity([response_embedding], topic_embeddings)[0]

    low_similarity_topics = [esg_topics[i] for i in np.argsort(similarities)[:3]]

    new_question = generate_focused_question(context, esg_categories, llm)

    return new_question, similarities


def iterative_extraction(rag_chain, initial_question, llm, max_iterations=3):
    current_question = initial_question
    all_responses = []
    all_contexts = []
    iteration_quality = []

    esg_topics = [topic for category in esg_categories.values() for topic in category]

    for i in range(max_iterations):
        print(f"\nイテレーション {i+1}/{max_iterations} 開始")
        start_time = time.time()

        print("  RAGチェーンを実行中...")
        response, context = rag_chain(current_question)
        all_responses.append(response)
        all_contexts.append(context)

        print("  新しい質問を生成中...")
        new_question, similarities = analyze_and_generate_new_question(
            response, context, initial_question, esg_topics, llm
        )
        current_question = new_question

        iteration_quality.append(np.mean(similarities))

        end_time = time.time()
        print(f"イテレーション {i+1} 完了 (所要時間: {end_time - start_time:.2f}秒)")

    final_response = extract_structured_info(all_responses[-1], esg_categories, llm)
    return final_response, iteration_quality, all_contexts


def calculate_similarity(text1, text2):
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    embedding1 = model.encode(text1)
    embedding2 = model.encode(text2)
    return cosine_similarity([embedding1], [embedding2])[0][0]


def calculate_factual_accuracy(extracted_info, reference_facts):
    extracted_keywords = set(extracted_info.lower().split())
    matched_facts = sum(
        1
        for fact in reference_facts
        if any(keyword in fact.lower() for keyword in extracted_keywords)
    )
    return matched_facts / len(reference_facts) if reference_facts else 0


def assess_investment_relevance(extracted_info, reference_impact):
    investment_keywords = set(reference_impact.lower().split())
    extracted_keywords = set(extracted_info.lower().split())
    relevance_score = len(investment_keywords.intersection(extracted_keywords)) / len(
        investment_keywords
    )
    return relevance_score


def evaluate_extraction_quality(structured_info, reference_data):
    scores = {"relevance": 0, "completeness": 0, "accuracy": 0, "investment_impact": 0}

    for category, subcategories in structured_info.items():
        for subcategory, info in subcategories.items():
            reference = reference_data[category][subcategory]

            scores["relevance"] += calculate_similarity(info, reference["content"])
            scores["completeness"] += len(info) / len(reference["content"])
            scores["accuracy"] += calculate_factual_accuracy(info, reference["facts"])
            scores["investment_impact"] += assess_investment_relevance(
                info, reference["impact"]
            )

    # Normalize scores
    total_items = sum(len(subcategories) for subcategories in esg_categories.values())
    for key in scores:
        scores[key] /= total_items

    return scores


def main(output_dir):
    print("ESG分析を開始します...")
    start_time = time.time()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_dir = "/root_nas05/home/2022/naoki/AI-Scientist/data/esg"
    os.makedirs(output_dir, exist_ok=True)

    print("準備済みデータを読み込んでいます...")
    vectorstore, meta_data = load_prepared_data(data_dir)

    print("Swallowモデルを設定しています...")
    llm = setup_swallow_model()

    print("改善されたRAGチェーンを設定しています...")
    rag_chain = setup_improved_rag_chain(vectorstore, llm)

    initial_question = "このレポートで議論されている主要なESG要因は何で、それらが投資判断にどのような影響を与える可能性がありますか？"
    print("反復抽出を開始します...")
    final_response, iteration_quality, all_contexts = iterative_extraction(
        rag_chain, initial_question, llm
    )

    print("結果をファイルに書き込んでいます...")
    with open(
        os.path.join(output_dir, f"esg_analysis_result_{current_time}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(final_response, f, ensure_ascii=False, indent=2)

    print("抽出結果を評価しています...")
    evaluation_results = evaluate_extraction_quality(final_response, reference_esg_data)

    results = {
        "structured_info": final_response,
        "evaluation_results": evaluation_results,
        "iteration_quality": [float(x) for x in iteration_quality],
    }

    print("評価結果をJSONファイルに書き込んでいます...")
    with open(
        os.path.join(output_dir, f"esg_analysis_results_{current_time}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("コンテキスト情報を保存しています...")
    with open(
        os.path.join(output_dir, f"esg_analysis_contexts_{current_time}.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        for i, context in enumerate(all_contexts):
            f.write(f"イテレーション {i+1} のコンテキスト:\n")
            f.write(context)
            f.write("\n\n" + "=" * 50 + "\n\n")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nESG分析が完了しました。総実行時間: {total_time:.2f}秒")

    # 実行時間も結果ファイルに追加
    with open(
        os.path.join(output_dir, f"esg_analysis_summary_{current_time}.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(f"ESG分析評価結果サマリー:\n\n")
        f.write(f"評価結果:\n")
        for metric, score in evaluation_results.items():
            f.write(f"  {metric}: {score:.4f}\n")
        f.write(f"\n反復品質:\n")
        for i, quality in enumerate(iteration_quality):
            f.write(f"  イテレーション {i+1}: {quality:.4f}\n")
        f.write(f"\n総実行時間: {total_time:.2f}秒\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ESG analysis")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()

    main(args.out_dir)
