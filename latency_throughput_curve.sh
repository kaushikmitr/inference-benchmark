#!/bin/bash

set -o xtrace

export IP=$IP
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

PYTHON="python3"
BENCHMARK_TIME_SECONDS=${BENCHMARK_TIME_SECONDS:-60}
MAX_NUM_PROMPTS=${MAX_NUM_PROMPTS:-1000}
SLEEP_TIME=${SLEEP_TIME:-0}
POST_BENCHMARK_SLEEP_TIME=${POST_BENCHMARK_SLEEP_TIME:-infinity}

IFS=',' read -ra RATES_1 <<< "$REQUEST_RATES_1"

if [[ -n "$REQUEST_RATES_2" ]]; then
  IFS=',' read -ra RATES_2 <<< "$REQUEST_RATES_2"
  if [[ ${#RATES_1[@]} -ne ${#RATES_2[@]} ]]; then
    echo "❌ REQUEST_RATES_1 and REQUEST_RATES_2 must have the same number of steps"
    exit 1
  fi
else
  RATES_2=()
fi

for i in "${!RATES_1[@]}"; do
  rate1=${RATES_1[$i]}
  rate2=${RATES_2[$i]:-}
  timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
  echo "🔁 Step $((i+1)) - Dataset1: $rate1 QPS${rate2:+, Dataset2: $rate2 QPS}"

  if [ "$rate1" == "0" ]; then
    rate1="inf"
    num_prompts1=$MAX_NUM_PROMPTS
  else
    num_prompts1=$(awk "BEGIN {print int($rate1 * $BENCHMARK_TIME_SECONDS)}")
  fi

  if [[ -n "$rate2" ]]; then
    if [ "$rate2" == "0" ]; then
      rate2="inf"
      num_prompts2=$MAX_NUM_PROMPTS
    else
      num_prompts2=$(awk "BEGIN {print int($rate2 * $BENCHMARK_TIME_SECONDS)}")
    fi
  fi

  PY_OPTS1=(
    "benchmark_serving.py"
    "--dataset=$DATASET1"
    "--file-prefix=$FILE_PREFIX_1"
    "--request-rate=$rate1"
    "--num-prompts=$num_prompts1"
    "--host=$IP"
    "--port=$PORT"
    "--tokenizer=$TOKENIZER"
    "--backend=$BACKEND"
    "--max-input-length=$INPUT_LENGTH"
    "--max-output-length=$OUTPUT_LENGTH"
    "--models=$MODELS"
    "--pm-namespace=$PM_NAMESPACE"
    "--pm-job=$PM_JOB"
    "--prometheus-port=$PROMETHEUS_PORT_1"
    "--save-json-results"
  )
  [[ "$MIN_INPUT_LENGTH" ]] && PY_OPTS1+=("--min-input-length=$MIN_INPUT_LENGTH")
  [[ "$MIN_OUTPUT_LENGTH" ]] && PY_OPTS1+=("--min-output-length=$MIN_OUTPUT_LENGTH")
  [[ "$OUTPUT_BUCKET" ]] && PY_OPTS1+=("--output-bucket=$OUTPUT_BUCKET")
  [[ "$TRAFFIC_SPLIT" ]] && PY_OPTS1+=("--traffic-split=$TRAFFIC_SPLIT")
  [[ "$SCRAPE_SERVER_METRICS" = "true" ]] && PY_OPTS1+=("--scrape-server-metrics")
  [[ "$SAVE_AGGREGATED_RESULT" = "true" ]] && PY_OPTS1+=("--save-aggregated-result")
  [[ "$STREAM_REQUEST" = "true" ]] && PY_OPTS1+=("--stream-request")
  [[ "$IGNORE_EOS" = "true" ]] && PY_OPTS1+=("--ignore-eos")
  [[ "$OUTPUT_BUCKET_FILEPATH" ]] && PY_OPTS1+=("--output-bucket-filepath=$OUTPUT_BUCKET_FILEPATH")
  [[ "$TCP_CONN_LIMIT" ]] && PY_OPTS1+=("--tcp-conn-limit=$TCP_CONN_LIMIT")
  [[ "$SPANNER_INSTANCE_ID" ]] && PY_OPTS1+=("--spanner-instance-id=$SPANNER_INSTANCE_ID")
  [[ "$SPANNER_DATABASE_ID" ]] && PY_OPTS1+=("--spanner-database-id=$SPANNER_DATABASE_ID")
  [[ "$TTFT_SLO_1" ]] && PY_OPTS1+=("--ttft-slo=$TTFT_SLO_1")
  [[ "$AVG_TPOT_SLO_1" ]] && PY_OPTS1+=("--avg-tpot-slo=$AVG_TPOT_SLO_1")
  [[ "$ENABLE_SLO_BASED_ROUTING_1" = "true" ]] && PY_OPTS1+=("--enable-slo-based-routing")

  output_file_1="latency-profile-${timestamp}-dataset1.txt"
  $PYTHON "${PY_OPTS1[@]}" > "$output_file_1" &
  pid1=$!

  if [[ -n "$DATASET2" && -n "$rate2" ]]; then
    PY_OPTS2=(
      "benchmark_serving.py"
      "--dataset=$DATASET2"
      "--file-prefix=$FILE_PREFIX_2"
      "--request-rate=$rate2"
      "--num-prompts=$num_prompts2"
      "--host=$IP"
      "--port=$PORT"
      "--tokenizer=$TOKENIZER"
      "--backend=$BACKEND"
      "--max-input-length=$INPUT_LENGTH"
      "--max-output-length=$OUTPUT_LENGTH"
      "--models=$MODELS"
      "--pm-namespace=$PM_NAMESPACE"
      "--pm-job=$PM_JOB"
      "--save-json-results"
      "--prometheus-port=$PROMETHEUS_PORT_2"
    )
    [[ "$MIN_INPUT_LENGTH" ]] && PY_OPTS2+=("--min-input-length=$MIN_INPUT_LENGTH")
    [[ "$MIN_OUTPUT_LENGTH" ]] && PY_OPTS2+=("--min-output-length=$MIN_OUTPUT_LENGTH")
    [[ "$OUTPUT_BUCKET" ]] && PY_OPTS2+=("--output-bucket=$OUTPUT_BUCKET")
    [[ "$TRAFFIC_SPLIT" ]] && PY_OPTS2+=("--traffic-split=$TRAFFIC_SPLIT")
    [[ "$SCRAPE_SERVER_METRICS" = "true" ]] && PY_OPTS2+=("--scrape-server-metrics")
    [[ "$SAVE_AGGREGATED_RESULT" = "true" ]] && PY_OPTS2+=("--save-aggregated-result")
    [[ "$STREAM_REQUEST" = "true" ]] && PY_OPTS2+=("--stream-request")
    [[ "$IGNORE_EOS" = "true" ]] && PY_OPTS2+=("--ignore-eos")
    [[ "$OUTPUT_BUCKET_FILEPATH" ]] && PY_OPTS2+=("--output-bucket-filepath=$OUTPUT_BUCKET_FILEPATH")
    [[ "$TCP_CONN_LIMIT" ]] && PY_OPTS2+=("--tcp-conn-limit=$TCP_CONN_LIMIT")
    [[ "$SPANNER_INSTANCE_ID" ]] && PY_OPTS2+=("--spanner-instance-id=$SPANNER_INSTANCE_ID")
    [[ "$SPANNER_DATABASE_ID" ]] && PY_OPTS2+=("--spanner-database-id=$SPANNER_DATABASE_ID")
    [[ "$TTFT_SLO_2" ]] && PY_OPTS2+=("--ttft-slo=$TTFT_SLO_2")
    [[ "$AVG_TPOT_SLO_2" ]] && PY_OPTS2+=("--avg-tpot-slo=$AVG_TPOT_SLO_2")
    [[ "$ENABLE_SLO_BASED_ROUTING_2" = "true" ]] && PY_OPTS2+=("--enable-slo-based-routing")

    output_file_2="latency-profile-${timestamp}-dataset2.txt"
    $PYTHON "${PY_OPTS2[@]}" > "$output_file_2" &
    pid2=$!
  fi

  wait $pid1
  [[ -n "$pid2" ]] && wait $pid2

  cat "$output_file_1"
  [[ -f "$output_file_2" ]] && cat "$output_file_2"
  echo "Sleeping for $SLEEP_TIME seconds..."
  sleep $SLEEP_TIME
done

export LPG_FINISHED="true"
sleep $POST_BENCHMARK_SLEEP_TIME
