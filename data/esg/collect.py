from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import requests
import logging
from datetime import datetime
import urllib3
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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


def create_session() -> requests.Session:
    """カスタマイズされたリトライ機能付きのセッションを作成"""
    session = requests.Session()

    # リトライ戦略の設定
    retry_strategy = Retry(
        total=5,  # リトライ回数を増やす
        backoff_factor=2,  # リトライ間の待機時間を長めに設定
        status_forcelist=[403, 408, 429, 500, 502, 503, 504],  # 403を追加
        allowed_methods=["GET", "HEAD", "OPTIONS"],  # 安全なメソッドのみリトライ
    )

    # カスタムヘッダーの追加
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/pdf,*/*",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "Referer": "https://www.google.com/",
        }
    )

    # HTTPアダプターの設定
    adapter = HTTPAdapter(
        max_retries=retry_strategy, pool_connections=10, pool_maxsize=10
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def get_target_companies() -> List[Report]:
    """収集対象の企業とレポートURLを定義"""
    reports = [
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
        Report(
            company_name="日産自動車",
            industry="自動車",
            url="https://www.nissan-global.com/JP/SUSTAINABILITY/LIBRARY/SR/2023/ASSETS/PDF/ESGDB23_J_All.pdf",
            year=2023,
            report_type="sustainability",
        ),
        Report(
            company_name="NTTデータ",
            industry="情報通信",
            url="https://www.nttdata.com/global/ja/-/media/nttdataglobal-ja/files/sustainability/report/library/2023/sr2023cb_all_b_jp_01.pdf",
            year=2023,
            report_type="sustainability",
        ),
        Report(
            company_name="野村総合研究所",
            industry="情報通信",
            url="https://www.nri.com/-/media/Corporate/jp/Files/PDF/sustainability/Sustainability_Book2023.pdf?la=ja-JP",
            year=2023,
            report_type="sustainability",
        ),
        Report(
            company_name="三菱総合研究所",
            industry="情報通信",
            url="https://www.mri.co.jp/sustainability/group-report/jdvs5f0000003zk3-att/MRIgroupreport2023_j.pdf",
            year=2023,
            report_type="sustainability",
        ),
        Report(
            company_name="セブン&アイ・ホールディングス",
            industry="小売",
            url="https://www.7andi.com/library/dbps_data/_template_/_res/sustainability/pdf/2023_all_02.pdf",
            year=2023,
            report_type="sustainability",
        ),
        Report(
            company_name="イオン",
            industry="小売",
            url="https://ssl4.eir-parts.net/doc/8267/ir_material_for_fiscal_ym22/143294/00.pdf",
            year=2023,
            report_type="integrated",
        ),
        Report(
            company_name="ドン・キホーテ",
            industry="小売",
            url="https://ppih.co.jp/ir/pdf/ar_2023.pdf",
            year=2023,
            report_type="integrated",
        ),
    ]
    return reports


def download_report(
    session: requests.Session, report: Report, output_dir: Path
) -> Optional[Path]:
    """レポートをダウンロードして保存"""
    try:
        # ダウンロード前に少し待機してサーバーへの負荷を軽減
        time.sleep(3)

        # 最初にHEADリクエストを送信してファイルの存在を確認
        head_response = session.head(report.url, timeout=30, verify=False)
        if head_response.status_code != 200:
            logging.warning(
                f"HEAD request failed for {report.company_name}: {head_response.status_code}"
            )

        # GETリクエストでファイルをダウンロード
        response = session.get(
            report.url,
            timeout=60,
            verify=False,
            allow_redirects=True,  # リダイレクトを許可
            stream=True,  # 大きなファイルのストリーミングダウンロード
        )

        if response.status_code == 200:
            # PDFファイルであることを確認
            content_type = response.headers.get("content-type", "").lower()
            if (
                "application/pdf" not in content_type
                and ".pdf" not in report.url.lower()
            ):
                logging.error(
                    f"Downloaded content for {report.company_name} is not a PDF"
                )
                return None

            # 保存用のファイル名を作成
            filename = f"{report.company_name}_{report.year}_{report.report_type}.pdf"
            output_path = output_dir / report.industry / filename

            # ディレクトリが存在しない場合は作成
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # ストリーミングでファイルを保存
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logging.info(f"Downloaded: {filename}")
            return output_path
        else:
            logging.error(
                f"Failed to download {report.company_name}: Status code {response.status_code}"
            )
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error downloading {report.company_name}: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error downloading {report.company_name}: {str(e)}")
        return None
    finally:
        # 接続を確実にクローズ
        if "response" in locals():
            response.close()


def main():
    # ログの設定
    setup_logging()

    # 出力ディレクトリの設定
    base_dir = Path("/root_nas05/home/2022/naoki/AI-Scientist/data/esg/raw")
    base_dir.mkdir(parents=True, exist_ok=True)

    # 対象企業のリストを取得
    reports = get_target_companies()

    # セッションの作成
    session = create_session()

    # ダウンロード実行
    successful_downloads = 0
    failed_downloads = 0

    for report in reports:
        try:
            if download_report(session, report, base_dir):
                successful_downloads += 1
            else:
                failed_downloads += 1
        except Exception as e:
            logging.error(f"Failed to process {report.company_name}: {str(e)}")
            failed_downloads += 1

    # セッションをクローズ
    session.close()

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
