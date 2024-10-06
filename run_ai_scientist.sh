#!/bin/bash

# 実行日時を取得
current_time=$(date "+%Y%m%d_%H%M%S")

# ログディレクトリ
log_dir="logs"

# ログディレクトリが存在しない場合は作成
if [ ! -d "$log_dir" ]; then
  mkdir "$log_dir"
fi

# 出力ファイル名
output_file="$log_dir/ai_scientist_${current_time}.log"
error_file="$log_dir/ai_scientist_${current_time}.err"

# コマンド
command="/root_nas05/home/2022/naoki/anaconda3/envs/ai_scientist/bin/python -u launch_scientist.py --model "gpt-4o-2024-08-06" --experiment nanoGPT --num-ideas 0 --gpus "0,1,2,3" --parallel 4 --improvement --skip-novelty-check"

# バックグラウンドで実行し、標準出力とエラー出力をファイルにリダイレクト
nohup $command > $output_file 2> $error_file &

echo "AI Scientist started in background. Output: $output_file, Error: $error_file"
