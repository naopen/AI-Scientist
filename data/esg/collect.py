from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import requests
import logging
from datetime import datetime
import urllib3

# SSLの警告を無効化（オプション）
urllib3.disable_warnings()


@dataclass
class Report:
    company_name: str
    industry: str
    url: str
    year: int
    report_type: str  # 'sustainability' or 'integrated'


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                f'data_collection_{datetime.now().strftime("%Y%m%d")}.log'
            ),
            logging.StreamHandler(),
        ],
    )


def get_target_companies() -> List[Report]:
    """収集対象の企業とレポートURLを定義"""
    reports = [
        # 輸送用機器（自動車）
        Report(
            company_name="アイシン",
            industry="自動車",
            url="https://www.aisin.com/jp/sustainability/report/pdf/aisin_ar2023_a3.pdf",
            year=2023,
            report_type="integrated",
        ),
        Report(
            company_name="ホンダ",
            industry="自動車",
            url="https://global.honda/jp/sustainability/report/pdf/2023/Honda-SR-2023-jp-all.pdf",
            year=2023,
            report_type="sustainability",
        ),
        Report(
            company_name="トヨタ自動車",
            industry="自動車",
            url="https://global.toyota/pages/global_toyota/ir/library/annual/2023_001_integrated_jp.pdf",
            year=2023,
            report_type="integrated",
        ),
        # 情報・通信業
        Report(
            company_name="日本電信電話",
            industry="情報通信",
            url="https://group.ntt/jp/ir/library/annual/pdf/integrated_report_23j_p.pdf",
            year=2023,
            report_type="integrated",
        ),
        Report(
            company_name="KDDI",
            industry="情報通信",
            url="https://www.kddi.com/extlib/files/corporate/ir/ir-library/sustainability-integrated-report/pdf/kddi_sir2023_j_p.pdf",
            year=2023,
            report_type="sustainability",
        ),
        Report(
            company_name="ソフトバンク",
            industry="情報通信",
            url="https://www.softbank.jp/corp/set/data/sustainability/documents/reports/pdf/sbkk_sustainability_report_2024.pdf",
            year=2023,
            report_type="sustainability",
        ),
        # 小売業
        Report(
            company_name="セブン&アイ・ホールディングス",
            industry="小売",
            url="https://www.7andi.com/library/dbps_data/_template_/_res/sustainability/pdf/2023_all_02.pdf",
            year=2023,
            report_type="sustainability",
        ),
        Report(
            company_name="ファーストリテイリング",
            industry="小売",
            url="https://www.fastretailing.com/jp/ir/library/pdf/ar2023.pdf",
            year=2023,
            report_type="sustainability",
        ),
        Report(
            company_name="マツキヨココカラ&カンパニー",
            industry="小売",
            url="https://www.matsukiyococokara.com/sustainability/integrated_report/pdf/MC&C_integrated_report_2023.pdf",
            year=2023,
            report_type="sustainability",
        ),
    ]
    return reports


def download_report(report: Report, output_dir: Path) -> Optional[Path]:
    """レポートをダウンロードして保存"""
    try:
        # verify=Falseを追加してSSL証明書の検証をスキップ
        response = requests.get(report.url, timeout=30, verify=False)
        if response.status_code == 200:
            # 保存用のファイル名を作成
            filename = f"{report.company_name}_{report.year}_{report.report_type}.pdf"
            output_path = output_dir / report.industry / filename

            # ディレクトリが存在しない場合は作成
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # ファイルを保存
            output_path.write_bytes(response.content)
            logging.info(f"Downloaded: {filename}")
            return output_path
        else:
            logging.error(
                f"Failed to download {report.company_name}: Status code {response.status_code}"
            )
            return None
    except Exception as e:
        logging.error(f"Error downloading {report.company_name}: {str(e)}")
        return None


def main():
    # ログの設定
    setup_logging()

    # 出力ディレクトリの設定
    base_dir = Path("/root_nas05/home/2022/naoki/AI-Scientist/data/esg/raw")
    base_dir.mkdir(parents=True, exist_ok=True)

    # 対象企業のリストを取得
    reports = get_target_companies()

    # ダウンロード実行
    successful_downloads = 0
    failed_downloads = 0

    for report in reports:
        if download_report(report, base_dir):
            successful_downloads += 1
        else:
            failed_downloads += 1

    # 結果の表示
    logging.info(
        f"""
    Download Summary:
    ----------------
    Total Reports: {len(reports)}
    Successful: {successful_downloads}
    Failed: {failed_downloads}
    """
    )


if __name__ == "__main__":
    main()
