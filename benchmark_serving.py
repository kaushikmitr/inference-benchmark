# Copyright 2024 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


r"""Benchmark LLM serving throughput and latency.
This script is for sending requests with prompts to LLM server and benchmark
the latency and throughput at various request rates.

It currently supports TGI, vLLM, Triton TensorRT-LLM and Saxml.
"""

import argparse
import asyncio
from datetime import datetime
import json
from locale import strcoll
import random
import requests
import time
from typing import AsyncGenerator, List, Optional, Tuple, Dict
from prometheus_client import start_http_server, Histogram, Gauge, Counter
import logging

import google.auth
import google.auth.transport.requests
from google.cloud import storage

import aiohttp
import numpy as np
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase

from google.protobuf.timestamp_pb2 import Timestamp



import os
import sys
import uuid
import traceback
from google.cloud import spanner
import math
from google.api_core import exceptions as gcp_exceptions

def safe_json_value(value, default=0.0):
    """Convert value to JSON-safe format, handling NaN and Infinity."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    return value

def extract_proto_fields(data, run_type):
    """Extract and structure relevant fields for Spanner insertion, including `run_type`."""

    config = {
        'model': data.get('config', {}).get('model', ''),
        'num_models': safe_json_value(data.get('config', {}).get('num_models', 0), 0),
        'model_server': data.get('config', {}).get('model_server', ''),
        'backend': data.get('dimensions', {}).get('backend', ''),
        'model_id': data.get('dimensions', {}).get('model_id', ''),
        'tokenizer_id': data.get('dimensions', {}).get('tokenizer_id', ''),
        'request_rate': safe_json_value(data.get('metrics', {}).get('request_rate', 0), 0),
        'benchmark_time': safe_json_value(data.get('metrics', {}).get('benchmark_time', 0), 0),
        'run_type': run_type
    }

    infrastructure = {
        'model_server': config['model_server'],
        'backend': config['backend'],
        'gpu_cache_usage_p90': safe_json_value(data.get('metrics', {}).get('server_metrics', {}).get('vllm:gpu_cache_usage_perc', {}).get('P90', 0.0)),
        'num_requests_waiting_p90': safe_json_value(data.get('metrics', {}).get('server_metrics', {}).get('vllm:num_requests_waiting', {}).get('P90', 0.0)),
        'gpu_cache_usage_mean': safe_json_value(data.get('metrics', {}).get('server_metrics', {}).get('vllm:gpu_cache_usage_perc', {}).get('Mean', 0.0)),
        'num_requests_waiting_mean': safe_json_value(data.get('metrics', {}).get('server_metrics', {}).get('vllm:num_requests_waiting', {}).get('Mean', 0.0)),
    }

    metrics = data.get('metrics', {})
    prompt_dataset = {
        'num_prompts_attempted': safe_json_value(metrics.get('num_prompts_attempted', 0), 0),
        'num_prompts_succeeded': safe_json_value(metrics.get('num_prompts_succeeded', 0), 0),
        'avg_input_len': safe_json_value(metrics.get('avg_input_len', 0.0)),
        'median_input_len': safe_json_value(metrics.get('median_input_len', 0.0)),
        'p90_input_len': safe_json_value(metrics.get('p90_input_len', 0.0)),
        'avg_output_len': safe_json_value(metrics.get('avg_output_len', 0.0)),
        'median_output_len': safe_json_value(metrics.get('median_output_len', 0.0)),
        'p90_output_len': safe_json_value(metrics.get('p90_output_len', 0.0))
    }

    summary_stats = {
        'p90_normalized_time_per_output_token_ms': safe_json_value(metrics.get('p90_normalized_time_per_output_token_ms', 0.0)),
        'avg_normalized_time_per_output_token_ms': safe_json_value(metrics.get('avg_normalized_time_per_output_token_ms', 0.0)),
        'throughput': safe_json_value(metrics.get('throughput', 0.0)),
        'input_tokens_per_sec': safe_json_value(metrics.get('input_tokens_per_sec', 0.0)),
        'benchmark_time': safe_json_value(metrics.get('benchmark_time', 0.0)),
        'date': data.get('dimensions', {}).get('date', ''),
        'avg_latency_ms': safe_json_value(metrics.get('avg_latency_ms', 0.0)),
        'median_latency_ms': safe_json_value(metrics.get('median_latency_ms', 0.0)),
        'p90_latency_ms': safe_json_value(metrics.get('p90_latency_ms', 0.0)),
        'p99_latency_ms': safe_json_value(metrics.get('p99_latency_ms', 0.0)),
        'time_per_output_token_seconds_p90': safe_json_value(data.get('metrics', {}).get('server_metrics', {}).get('vllm:time_per_output_token_seconds', {}).get('P90', 0.0)),
        'time_to_first_token_seconds_p90': safe_json_value(data.get('metrics', {}).get('server_metrics', {}).get('vllm:time_to_first_token_seconds', {}).get('P90', 0.0)),
        'time_per_output_token_seconds_mean': safe_json_value(data.get('metrics', {}).get('server_metrics', {}).get('vllm:time_per_output_token_seconds', {}).get('Mean', 0.0)),
        'time_to_first_token_seconds_mean': safe_json_value(data.get('metrics', {}).get('server_metrics', {}).get('vllm:time_to_first_token_seconds', {}).get('Mean', 0.0)),
    }

    return config, infrastructure, prompt_dataset, summary_stats
  
def clean_for_json(obj):
    """Recursively clean an object for JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    elif obj is None:
        return 0.0
    else:
        return obj
      
def upload_to_spanner_batch_with_retry(instance_id, database_id, json_files, gcs_base_uri, run_type, max_retries=3):
    """
    Upload JSON files to Spanner in batches with retry logic.
    More efficient but fails entire batch if any single file has issues.
    """
    spanner_client = spanner.Client()
    instance = spanner_client.instance(instance_id)
    database = instance.database(database_id)

    print(f"📊 Uploading {len(json_files)} JSON files to Spanner with run_type='{run_type}'...")

    retry_count = 0
    success = False
    processed_files = []
    
    while retry_count <= max_retries and not success:
        try:
            processed_files = []  # Reset on each retry
            
            with database.batch() as batch:
                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)

                        config, infra, prompt, stats = extract_proto_fields(data, run_type)
                        filename = os.path.basename(json_file)
                        gcs_uri = f"{gcs_base_uri}/{filename}"
                        latency_profile_id = str(uuid.uuid4())

                        # Test JSON serialization before inserting
                        try:
                            config_clean = clean_for_json(config)
                            infra_clean = clean_for_json(infra)
                            prompt_clean = clean_for_json(prompt)
                            stats_clean = clean_for_json(stats)
                
                            config_json = json.dumps(config_clean)
                            infra_json = json.dumps(infra_clean)
                            prompt_json = json.dumps(prompt_clean)
                            stats_json = json.dumps(stats_clean)
                        except (TypeError, ValueError) as json_error:
                            print(f"❌ JSON serialization failed for {json_file}: {json_error}")
                            continue

                        batch.insert(
                            table='LatencyProfiles',
                            columns=['Id', 'Config', 'Infrastructure', 'PromptDataset', 'SummaryStats', 'GcsUri', 'InsertedAt'],
                            values=[
                                (latency_profile_id, config_json, infra_json, prompt_json, stats_json, gcs_uri, spanner.COMMIT_TIMESTAMP)
                            ]
                        )

                        if 'core_deployment_artifacts' in data or 'extension_deployment_artifacts' in data:
                            core_json = json.dumps(data.get('core_deployment_artifacts', {}))
                            ext_json = json.dumps(data.get('extension_deployment_artifacts', {}))
                            batch.insert(
                                table='DeploymentArtifacts',
                                columns=['Id', 'LatencyProfileId', 'CoreDeploymentArtifacts', 'ExtensionDeploymentArtifacts'],
                                values=[
                                    (str(uuid.uuid4()), latency_profile_id, core_json, ext_json)
                                ]
                            )
                        
                        processed_files.append((json_file, latency_profile_id))
                        
                    except Exception as e:
                        print(f"❌ Failed to process {json_file}: {e}")
                        continue
            
            # If we get here, the batch committed successfully
            for json_file, profile_id in processed_files:
                print(f"✅ {json_file} uploaded (ID: {profile_id})")
            success = True
            
        except (gcp_exceptions.DeadlineExceeded, 
                gcp_exceptions.ServiceUnavailable,
                gcp_exceptions.InternalServerError,
                gcp_exceptions.TooManyRequests) as retryable_error:
            retry_count += 1
            if retry_count <= max_retries:
                wait_time = (2 ** retry_count) + random.uniform(0, 1)
                print(f"⚠️  Batch upload failed (attempt {retry_count}/{max_retries}): {retryable_error}")
                print(f"⏳ Retrying entire batch in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"❌ Batch upload failed after {max_retries} retries: {retryable_error}")
                
        except Exception as batch_error:
            retry_count += 1
            if retry_count <= max_retries:
                wait_time = (2 ** retry_count) + random.uniform(0, 1)
                print(f"⚠️  Unexpected batch error (attempt {retry_count}/{max_retries}): {batch_error}")
                print(f"⏳ Retrying entire batch in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"❌ Batch upload failed after {max_retries} retries: {batch_error}")
                traceback.print_exc()

    if success:
        print("✅ All files uploaded successfully.")
    else:
        print("❌ Upload process failed after all retries.")


def calculate_prediction_errors(actual_values, predicted_values):
    """Calculate MAPE, RMSE, and MAE between actual and predicted values."""
    if not actual_values or not predicted_values or len(actual_values) != len(predicted_values):
        return {
            'mape': 0.0,
            'rmse': 0.0,
            'mae': 0.0,
            'count': 0
        }
    
    # Filter out pairs where either value is 0 or negative (to avoid division by zero in MAPE)
    valid_pairs = [(a, p) for a, p in zip(actual_values, predicted_values) if a > 0 and p > 0]
    
    if not valid_pairs:
        return {
            'mape': 0.0,
            'rmse': 0.0,
            'mae': 0.0,
            'count': 0
        }
    
    actual_clean = [pair[0] for pair in valid_pairs]
    predicted_clean = [pair[1] for pair in valid_pairs]
    
    # Calculate errors
    absolute_errors = [abs(a - p) for a, p in valid_pairs]
    percentage_errors = [abs((a - p) / a) * 100 for a, p in valid_pairs]
    squared_errors = [(a - p) ** 2 for a, p in valid_pairs]
    
    mape = np.mean(percentage_errors)
    mae = np.mean(absolute_errors)
    rmse = np.sqrt(np.mean(squared_errors))
    
    return {
        'mape': mape,
        'rmse': rmse,
        'mae': mae,
        'count': len(valid_pairs)
    }

def get_prediction_error_stats(name, actual_values, predicted_values):
    """Get prediction error statistics and format them for output."""
    errors = calculate_prediction_errors(actual_values, predicted_values)
    
    if errors['count'] > 0:
        print(f"Prediction errors for {name}:")
        print(f"  MAPE: {errors['mape']:.2f}%")
        print(f"  RMSE: {errors['rmse']:.2f}")
        print(f"  MAE: {errors['mae']:.2f}")
        print(f"  Valid predictions: {errors['count']}")
    else:
        print(f"No valid prediction pairs for {name}")
    
    return {
        f'{name}_prediction_mape': errors['mape'],
        f'{name}_prediction_rmse': errors['rmse'],
        f'{name}_prediction_mae': errors['mae'],
        f'{name}_prediction_count': errors['count']
    }

MIN_SEQ_LEN = 4
NEW_TEXT_KEY = "\nOutput:\n"


# Prometheus Metrics
prompt_length_metric = Histogram("LatencyProfileGenerator:prompt_length", "Input prompt length", buckets=[2**i for i in range(1, 16)])
response_length_metric = Histogram("LatencyProfileGenerator:response_length", "Response length", buckets=[2**i for i in range(1, 16)])
normalized_time_per_output_token_metric = Histogram('LatencyProfileGenerator:normalized_time_per_output_token_ms', 'Request time over total number of tokens (including first token) (ms)', buckets=[2**i for i in range(1, 16)])
tpot_metric = Histogram('LatencyProfileGenerator:time_per_output_token_ms', 'Time per output token per request (excluding first token) (ms)', buckets=[2**i for i in range(1, 16)])
ttft_metric = Histogram('LatencyProfileGenerator:time_to_first_token_ms', 'Time to first token per request (ms)', buckets=[2**i for i in range(1, 16)])
active_requests_metric = Gauge('LatencyProfileGenerator:active_requests', 'How many requests actively being processed')
total_request_count = Counter('LatencyProfileGenerator:request_count', 'How many total requests have been sent')

# Additional metrics for client vs server tracking
client_ttft_metric = Histogram('LatencyProfileGenerator:client_time_to_first_token_ms', 'Client-measured time to first token (ms)', buckets=[2**i for i in range(1, 16)])
server_ttft_metric = Histogram('LatencyProfileGenerator:server_time_to_first_token_ms', 'Server-reported time to first token (ms)', buckets=[2**i for i in range(1, 16)])
client_tpot_metric = Histogram('LatencyProfileGenerator:client_time_per_output_token_ms', 'Client-measured time per output token (ms)', buckets=[2**i for i in range(1, 16)])
server_tpot_metric = Histogram('LatencyProfileGenerator:server_time_per_output_token_ms', 'Server-reported time per output token (ms)', buckets=[2**i for i in range(1, 16)])

# Singleton class to track requests for QPS counting and calculation.
class AsyncRequestCounter:
  _instance = None
  _lock = asyncio.Lock()

  async def __new__(cls, target_requests=None, *args, **kwargs):
    async with cls._lock:
      if not cls._instance:
        cls._instance = super().__new__(cls)
        cls._instance._count = 0
        cls._instance._start_time = time.time()
        cls._instance._target_requests = target_requests
    return cls._instance
  
  async def increment(self):
    async with self._lock:
      self._count += 1
      if self._count == self._target_requests:
        self._end_time = time.time()
  
  async def get_qps(self):
    return self._count / (self._end_time - self._start_time)


# Add trace config for monitoring in flight requests
async def on_request_start(session, trace_config_ctx, params):
    active_requests_metric.inc()
    total_request_count.inc()
    counter = await AsyncRequestCounter()
    await counter.increment()

async def on_request_end(session, trace_config_ctx, params):
    active_requests_metric.dec()

trace_config = aiohttp.TraceConfig()
trace_config.on_request_start.append(on_request_start)
trace_config.on_request_end.append(on_request_end)

# Google Cloud Storage Client
gcs_client = None
gcs_bucket = None

def get_filtered_dataset(
    dataset_path: str,
    max_input_len: int,
    max_output_len: int,
    min_input_len: int,
    min_output_len: int,
    tokenizer: PreTrainedTokenizerBase,
    use_dummy_text: bool,
) -> List[Tuple[str, int, int]]:
  """Samples requests from the dataset or creates dummy requests."""
  if use_dummy_text:
    dummy_prompt_token_ids = [0] * max_input_len
    dummy_prompt = tokenizer.decode(dummy_prompt_token_ids)
    return [(
          dummy_prompt,
          max_input_len,
          max_output_len,
    )]

  # Load the dataset.
  with open(dataset_path) as f:
    dataset = json.load(f)
  # Filter out the conversations with less than 2 turns.
  dataset = [data for data in dataset if len(data["conversations"]) >= 2]
  # Only keep the first two turns of each conversation.
  dataset = [
      (data["conversations"][0]["value"], data["conversations"][1]["value"])
      for data in dataset
  ]

  # Tokenize the prompts and completions.
  prompts = [prompt for prompt, _ in dataset]
  prompt_token_ids = tokenizer(prompts).input_ids
  completions = [completion for _, completion in dataset]
  completion_token_ids = tokenizer(completions).input_ids
  tokenized_dataset = []
  for i in range(len(dataset)):
    output_len = len(completion_token_ids[i])
    tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

  # Filter out too long sequences.
  filtered_dataset: List[Tuple[str, int, int]] = []
  for prompt, prompt_token_ids, output_len in tokenized_dataset:
    prompt_len = len(prompt_token_ids)
    if prompt_len < min_input_len or output_len < min_output_len:
      # Prune too short sequences.
      # This is because TGI causes errors when the input or output length
      # is too short.
      continue
    if prompt_len > max_input_len or output_len > max_output_len:
      # Prune too long sequences.
      continue
    filtered_dataset.append((prompt, prompt_len, output_len))

  return filtered_dataset

async def generate_next_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
  """Gets request async."""
  while True:
    request = random.choice(input_requests)
    yield request

    if request_rate == float("inf"):
      # If the request rate is infinity, then we don't need to wait.
      continue
    # Sample the request interval from the exponential distribution.
    interval = np.random.exponential(1.0 / request_rate)
    # The next request will be sent after the interval.
    await asyncio.sleep(interval)

def init_errors_map() -> Dict[str, int]:
  errors = {
    "ClientConnectorError": 0,
    "TimeoutError": 0,
    "ContentTypeError": 0,
    "ClientOSError": 0,
    "ServerDisconnectedError": 0,
    "unknown_error": 0,
    "HTTP429": 0,
  }
  return errors

async def send_stream_request(
    backend: str,
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    ignore_eos: bool,
    best_of: int,
    use_beam_search: bool,
    top_k: int,
    tokenizer: PreTrainedTokenizerBase,
    sax_model: str,
    model: str,
    timeout: float,
    max_conn: int,
    ttft_slo: Optional[float] = None,
    avg_tpot_slo: Optional[float] = None,
    enable_slo_based_routing: bool = False,
) -> Tuple[
      Optional[Tuple[int, int, float]],  # (prompt_len, output_len, latency) or None
      float,   # server_ttft_ms
      float,   # client_ttft_ms
      float,   # server_avg_tpot_ms
      float,   # client_avg_tpot_ms
      float,   # predicted_ttft_ms
      float,   # predicted_tpot_ms
      Dict[str, int]
]:
    """Send streaming request and track both server and client-side metrics."""
    request_start_time_ms = 1000 * time.time()
    errors = init_errors_map()

    # Initialize metrics
    server_ttft_ms = 0.0
    client_ttft_ms = 0.0
    server_avg_tpot_ms = 0.0
    client_avg_tpot_ms = 0.0
    predicted_ttft_ms = 0.0
    predicted_tpot_ms = 0.0
    
    # Client-side timing tracking
    start_perf_ms = 1000 * time.perf_counter()
    first_token_time_ms = None
    token_times = []
    total_tokens_received = 0
    last_usage = None
    accumulated_text = ""

    # Build headers & payload
    headers = {"User-Agent": "Benchmark Client"}
    if ttft_slo is not None and enable_slo_based_routing:
        headers["ttft_slo"] = f"{ttft_slo:.6f}"
    if avg_tpot_slo is not None and enable_slo_based_routing:
        headers["avg_tpot_slo"] = f"{avg_tpot_slo:.6f}"

    pload = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "best_of": best_of,
        "use_beam_search": use_beam_search,
        "temperature": 0.0 if use_beam_search else 1.0,
        "top_p": 1.0,
        "max_tokens": output_len,
        "ignore_eos": ignore_eos,
        "stream": True,
        "stream_options": {"include_usage": "true"}
    }

    timeout_cfg = aiohttp.ClientTimeout(total=timeout)
    
    try:
        async with aiohttp.ClientSession(
            timeout=timeout_cfg,
            trust_env=True,
            trace_configs=[trace_config],
            connector=aiohttp.TCPConnector(limit=max_conn),
        ) as session:
            async with session.post(api_url, headers=headers, json=pload, ssl=False) as resp:
                if resp.status == 429:
                    errors["HTTP429"] += 1
                    return None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, errors
                if resp.status != 200:
                    errors["unknown_error"] += 1
                    return None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, errors
                
                async for raw_chunk in resp.content.iter_chunks():
                    chunk = raw_chunk[0].strip()
                    if not chunk:
                        continue

                    current_time_ms = 1000 * time.perf_counter()
                    
                    # Decode chunk
                    try:
                        text = chunk.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    
                    # Handle multiple data lines in one chunk
                    lines = text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Remove SSE prefix
                        if line.startswith("data: "):
                            payload = line[6:]
                        else:
                            payload = line

                        if payload == "[DONE]":
                            break

                        # Parse the JSON event
                        try:
                            evt = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        # Track first token timing (client-side)
                        choices = evt.get("choices", [])
                        if choices and first_token_time_ms is None:
                            # Check if this chunk contains actual content
                            current_text = choices[0].get("text", "")
                            if current_text.strip():
                                first_token_time_ms = current_time_ms
                                client_ttft_ms = first_token_time_ms - start_perf_ms
                        
                        # Track tokens for TPOT calculation
                        if choices and first_token_time_ms is not None:
                            current_text = choices[0].get("text", "")
                            if current_text:
                                # Add to accumulated text and count new tokens
                                #new_text = current_text[len(accumulated_text):] if current_text.startswith(accumulated_text) else current_text
                                if current_text.strip():
                                    token_times.append(current_time_ms)
                                    total_tokens_received += 1
                                    accumulated_text += current_text

                        # Extract server-provided usage metrics (usually in final chunk)
                        usage = evt.get("usage")
                        if isinstance(usage, dict) and usage:
                            last_usage = usage

    except aiohttp.client_exceptions.ClientConnectorError:
        errors["ClientConnectorError"] += 1
        return None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, errors
    except asyncio.TimeoutError:
        errors["TimeoutError"] += 1
        return None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, errors
    except aiohttp.client_exceptions.ClientOSError:
        errors["ClientOSError"] += 1
        return None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, errors
    except aiohttp.client_exceptions.ContentTypeError:
        errors["ContentTypeError"] += 1
        return None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, errors
    except aiohttp.client_exceptions.ServerDisconnectedError:
        errors["ServerDisconnectedError"] += 1
        return None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, errors
    except Exception as e:
        print(f"Unexpected error in send_stream_request: {e}")
        errors["unknown_error"] += 1
        return None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, errors

    # Calculate final metrics
    request_end_time_ms = 1000 * time.time()
    total_latency_ms = request_end_time_ms - request_start_time_ms

    # Calculate client-side TPOT if we have multiple token arrivals
    if len(token_times) > 1:
        # Calculate time between consecutive token chunks
        inter_token_times = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
        client_avg_tpot_ms = np.mean(inter_token_times) if inter_token_times else 0.0
    elif first_token_time_ms is not None and len(token_times) == 1:
        # Only one token chunk received
        client_avg_tpot_ms = max(0.0, (1000 * time.perf_counter() - first_token_time_ms))
    else:
        client_avg_tpot_ms = 0.0

    # Extract server-provided metrics from your specific format
    if last_usage is not None:
        server_ttft_ms = float(last_usage.get("ttft_ms", 0.0))
        predicted_ttft_ms = float(last_usage.get("predicted_ttft_ms", 0.0))
        server_avg_tpot_ms = float(last_usage.get("avg_tpot_ms", 0.0))
        predicted_tpot_ms = float(last_usage.get("avg_predicted_tpot_ms", 0.0))


            
        
        # Use server-provided token counts if available
        completion_tokens = last_usage.get("completion_tokens", 0)
        if completion_tokens > 0:
            actual_output_len = completion_tokens
        else:
            actual_output_len = total_tokens_received if total_tokens_received > 0 else output_len
    else:
        # Fallback to client measurements
        actual_output_len = total_tokens_received if total_tokens_received > 0 else output_len


    # Record metrics (prefer server metrics when available, fall back to client metrics)
    ttft_to_record = server_ttft_ms if server_ttft_ms > 0 else client_ttft_ms
    tpot_to_record = server_avg_tpot_ms if server_avg_tpot_ms > 0 else client_avg_tpot_ms

    if ttft_to_record > 0:
        ttft_metric.observe(ttft_to_record)
    if tpot_to_record > 0:
        tpot_metric.observe(tpot_to_record)
    
    # Record separate client and server metrics
    if client_ttft_ms > 0:
        client_ttft_metric.observe(client_ttft_ms)
    if server_ttft_ms > 0:
        server_ttft_metric.observe(server_ttft_ms)
    if client_avg_tpot_ms > 0:
        client_tpot_metric.observe(client_avg_tpot_ms)
    if server_avg_tpot_ms > 0:
        server_tpot_metric.observe(server_avg_tpot_ms)

    normalized_time_per_output_token_metric.observe(total_latency_ms / actual_output_len)
    prompt_length_metric.observe(prompt_len)
    response_length_metric.observe(actual_output_len)

    return (
        (prompt_len, actual_output_len, total_latency_ms),
        server_ttft_ms,
        client_ttft_ms,
        server_avg_tpot_ms,
        client_avg_tpot_ms,
        predicted_ttft_ms,
        predicted_tpot_ms,
        errors
    )

async def send_request(
    backend: str,
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    ignore_eos: bool,
    best_of: int,
    use_beam_search: bool,
    top_k: int,
    tokenizer: PreTrainedTokenizerBase,
    sax_model: str,
    model: str,
    timeout: float,
    max_conn: int,
) -> Tuple[
      Optional[Tuple[int, int, float]],  # (prompt_len, output_len, latency) or None
      float,   # server_ttft_ms
      float,   # client_ttft_ms (always 0 for non-streaming)
      float,   # server_avg_tpot_ms
      float,   # client_avg_tpot_ms (always 0 for non-streaming)
      float,   # predicted_ttft_ms
      float,   # predicted_tpot_ms
      Dict[str, int]
]:
    """Send non-streaming request with consistent error handling."""
    request_start_time_ms = 1000 * time.time()
    errors = init_errors_map()

    headers = {"User-Agent": "Benchmark Client"}
    
    # Build payload based on backend
    if backend == "vllm":
        pload = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": ignore_eos,
            "stream": False,
        }
    elif backend == "tgi":
        pload = {
            "inputs": prompt,
            "parameters": {
                "best_of": best_of,
                "max_new_tokens": output_len,
                "do_sample": True,
            },
            "stream_options": {"include_usage": "true"}
        }
    elif backend == "sax":
        pload = {
            "model": sax_model,
            "prompt": prompt,
            "max_tokens": output_len,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_k": top_k,
        }
    elif backend == "jetstream":
        pload = {
            "prompt": prompt,
            "max_tokens": output_len,
            "temperature": 0.0 if use_beam_search else 1.0,
        }
    else:
        # Default payload for other backends
        pload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": output_len,
            "stream": False,
        }

    timeout_cfg = aiohttp.ClientTimeout(total=timeout)
    output = None
    
    try:
        async with aiohttp.ClientSession(
            timeout=timeout_cfg,
            trust_env=True,
            trace_configs=[trace_config],
            connector=aiohttp.TCPConnector(limit=max_conn),
        ) as session:
            async with session.post(api_url, headers=headers, json=pload, ssl=False) as resp:
                
                if resp.status != 200:
                    errors["unknown_error"] += 1
                    return None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, errors
                
                output = await resp.json()
                
                if "error" in output:
                    errors["unknown_error"] += 1
                    return None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, errors
                    
    except aiohttp.client_exceptions.ClientConnectorError:
        errors["ClientConnectorError"] += 1
        return None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, errors
    except asyncio.TimeoutError:
        errors["TimeoutError"] += 1
        return None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, errors
    except aiohttp.client_exceptions.ClientOSError:
        errors["ClientOSError"] += 1
        return None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, errors
    except aiohttp.client_exceptions.ContentTypeError:
        errors["ContentTypeError"] += 1
        return None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, errors
    except aiohttp.client_exceptions.ServerDisconnectedError:
        errors["ServerDisconnectedError"] += 1
        return None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, errors
    except Exception as e:
        print(f"Unexpected error in send_request: {e}")
        errors["unknown_error"] += 1
        return None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, errors

    request_end_time_ms = 1000 * time.time()
    total_latency_ms = request_end_time_ms - request_start_time_ms

    # Extract server-provided usage metrics
    usage = output.get("usage", {}) or {}
    server_ttft_ms = float(usage.get("ttft_ms", 0.0))
    predicted_ttft_ms = float(usage.get("predicted_ttft_ms", 0.0))
    server_avg_tpot_ms = float(usage.get("avg_tpot_ms", 0.0))
    predicted_tpot_ms = float(usage.get("avg_predicted_tpot_ms", 0.0))

    # Calculate actual output length
    text = ""
    if backend == "vllm":
        choices = output.get("choices", [])
        if choices:
            text = choices[0].get("text", "")
    elif backend == "tgi":
        # TGI response format
        generated_text = output.get("generated_text", "")
        text = generated_text
    elif backend == "sax":
        # SAX response format
        text = output.get("text", "")
    elif backend == "jetstream":
        # Jetstream response format
        choices = output.get("choices", [])
        if choices:
            text = choices[0].get("text", "")
    # Add other backends as needed
    
    if text:
        output_token_ids = tokenizer(text).input_ids
        actual_output_len = len(output_token_ids)
    else:
        actual_output_len = output_len

    # Record metrics (only server metrics available for non-streaming)
    if server_ttft_ms > 0:
        ttft_metric.observe(server_ttft_ms)
        server_ttft_metric.observe(server_ttft_ms)
    if server_avg_tpot_ms > 0:
        tpot_metric.observe(server_avg_tpot_ms)
        server_tpot_metric.observe(server_avg_tpot_ms)

    normalized_time_per_output_token_metric.observe(total_latency_ms / actual_output_len)
    prompt_length_metric.observe(prompt_len)
    response_length_metric.observe(actual_output_len)

    return (
        (prompt_len, actual_output_len, total_latency_ms),
        server_ttft_ms,
        0.0,  # client_ttft_ms (not available for non-streaming)
        server_avg_tpot_ms,
        0.0,  # client_avg_tpot_ms (not available for non-streaming)
        predicted_ttft_ms,
        predicted_tpot_ms,
        errors
    )


async def run_single_request(args: argparse.Namespace, api_url: str, tokenizer: PreTrainedTokenizerBase,
                               prompt: str, prompt_len: int, output_len: int, chosen_model: str) -> Tuple[str, Tuple]:
    """Run a single request with proper error handling."""
    if args.stream_request:
        result = await send_stream_request(
            args.backend, api_url, prompt, prompt_len, output_len, args.ignore_eos,
            args.best_of, args.use_beam_search, args.top_k, tokenizer, args.sax_model,
            chosen_model, args.request_timeout, args.tcp_conn_limit, args.ttft_slo, args.avg_tpot_slo, args.enable_slo_based_routing)
    else:
        result = await send_request(
            args.backend, api_url, prompt, prompt_len, output_len, args.ignore_eos,
            args.best_of, args.use_beam_search, args.top_k, tokenizer, args.sax_model,
            chosen_model, args.request_timeout, args.tcp_conn_limit)
    return chosen_model, result


async def benchmark(
    args: argparse.Namespace, 
    api_url: str,
    tokenizer: PreTrainedTokenizerBase,
    models: List[str],
    traffic_split: List[float],
) -> None:
    """Runs benchmark requests with improved metrics tracking."""
    input_requests = get_filtered_dataset(
        args.dataset, args.max_input_length, args.max_output_length, 
        args.min_input_length, args.min_input_length, tokenizer, args.use_dummy_text)
    
    if traffic_split is None:
        traffic_split = [1.0 / len(models)] * len(models)
    if len(models) != len(traffic_split):
        raise ValueError("The number of models and traffic split values must match")
    total_weight = sum(traffic_split)
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"Traffic split must sum to 1.0, but got {total_weight}")
    
    models_dict = dict(zip(models, traffic_split))
    model_names = list(models_dict.keys())
    model_weights = list(models_dict.values())

    benchmark_start_time_sec = time.time()
    await AsyncRequestCounter(args.num_prompts)
    tasks: List[asyncio.Task] = []
    prompts_sent = 0
    
    async for request in generate_next_request(input_requests, args.request_rate):
        if prompts_sent >= args.num_prompts:
            break
        prompt, prompt_len, output_len = request
        chosen_model = random.choices(model_names, weights=model_weights)[0]
        task = asyncio.create_task(run_single_request(args, api_url, tokenizer, prompt, prompt_len, output_len, chosen_model))
        tasks.append(task)
        prompts_sent += 1

    results = await asyncio.gather(*tasks)

    # Initialize results tracking with expanded metrics including prediction pairs
    overall_results = {
        "latencies": [], 
        "server_ttfts": [], "client_ttfts": [],
        "server_tpots": [], "client_tpots": [],
        "predicted_ttfts": [], "predicted_tpots": [],
        "server_predicted_ttft_pairs": [],  # New: (server, predicted) pairs
        "server_predicted_tpot_pairs": [],  # New: (server, predicted) pairs
        "errors": init_errors_map(),
        "slo_met_count":    0,
    }
    
    per_model_results: Dict[str, Dict[str, List]] = {}
    for model in model_names:
        per_model_results[model] = {
            "latencies": [], 
            "server_ttfts": [], "client_ttfts": [],
            "server_tpots": [], "client_tpots": [],
            "predicted_ttfts": [], "predicted_tpots": [],
            "server_predicted_ttft_pairs": [],  # New: (server, predicted) pairs
            "server_predicted_tpot_pairs": [],  # New: (server, predicted) pairs
            "errors": init_errors_map()
        }

    # Process results with improved error handling
    for chosen_model, res in results:
        if res is None or res[0] is None:
            # Handle failed requests - count errors but don't skip entirely
            if res is not None and len(res) > 7:
                errors = res[7]
                for k, v in errors.items():
                    overall_results["errors"][k] += v
                    per_model_results[chosen_model]["errors"][k] += v
            continue
            
        latency, server_ttft, client_ttft, server_tpot, client_tpot, pred_ttft, pred_tpot, errors = res
        
        # Count any errors
        for k, v in errors.items():
            overall_results["errors"][k] += v
            per_model_results[chosen_model]["errors"][k] += v
        
        # Process successful request metrics
        if latency is not None:
            if args.stream_request:  # Only track SLO for streaming requests
                ttft_slo_met = args.ttft_slo is None or (server_ttft > 0 and server_ttft <= args.ttft_slo)
                tpot_slo_met = args.avg_tpot_slo is None or (server_tpot > 0 and server_tpot <= args.avg_tpot_slo)
    
            if ttft_slo_met and tpot_slo_met:
                    overall_results["slo_met_count"] += 1
    
            overall_results["latencies"].append(latency)
            per_model_results[chosen_model]["latencies"].append(latency)
            
            # Track all timing metrics
            if server_ttft > 0:
                overall_results["server_ttfts"].append(server_ttft)
                per_model_results[chosen_model]["server_ttfts"].append(server_ttft)
            if client_ttft > 0:
                overall_results["client_ttfts"].append(client_ttft)
                per_model_results[chosen_model]["client_ttfts"].append(client_ttft)
            if server_tpot > 0:
                overall_results["server_tpots"].append(server_tpot)
                per_model_results[chosen_model]["server_tpots"].append(server_tpot)
            if client_tpot > 0:
                overall_results["client_tpots"].append(client_tpot)
                per_model_results[chosen_model]["client_tpots"].append(client_tpot)
            if pred_ttft > 0:
                overall_results["predicted_ttfts"].append(pred_ttft)
                per_model_results[chosen_model]["predicted_ttfts"].append(pred_ttft)
            if pred_tpot > 0:
                overall_results["predicted_tpots"].append(pred_tpot)
                per_model_results[chosen_model]["predicted_tpots"].append(pred_tpot)
            
            # NEW: Track prediction pairs for error calculation
            if server_ttft > 0 and pred_ttft > 0:
                overall_results["server_predicted_ttft_pairs"].append((server_ttft, pred_ttft))
                per_model_results[chosen_model]["server_predicted_ttft_pairs"].append((server_ttft, pred_ttft))

            if server_tpot > 0 and pred_tpot > 0:
                overall_results["server_predicted_tpot_pairs"].append((server_tpot, pred_tpot))
                per_model_results[chosen_model]["server_predicted_tpot_pairs"].append((server_tpot, pred_tpot))


    benchmark_duration_sec = time.time() - benchmark_start_time_sec
    
    # Print results with both server and client metrics plus prediction errors
    await print_and_save_result(
        args, benchmark_duration_sec, prompts_sent, "weighted",
        overall_results["latencies"], 
        overall_results["server_ttfts"], overall_results["client_ttfts"],
        overall_results["predicted_ttfts"], 
        overall_results["server_tpots"], overall_results["client_tpots"],
        overall_results["predicted_tpots"],
        overall_results["server_predicted_ttft_pairs"],  # New parameter
        overall_results["server_predicted_tpot_pairs"],  # New parameter
        overall_results["errors"], 
        overall_results["slo_met_count"],
        spanner_upload=True, server_metrics_scrape=True
    )
    
    # Print per-model results
    for model, data in per_model_results.items():
        await print_and_save_result(
            args, benchmark_duration_sec, len(data["latencies"]), model,
            data["latencies"], 
            data["server_ttfts"], data["client_ttfts"],
            data["predicted_ttfts"],
            data["server_tpots"], data["client_tpots"], 
            data["predicted_tpots"],
            data["server_predicted_ttft_pairs"],  # New parameter
            data["server_predicted_tpot_pairs"],  # New parameter
            data["errors"]
        )


def save_json_results(args: argparse.Namespace, benchmark_result, server_metrics, model, errors, spanner_upload: bool = False):
  # Setup
  start_dt_proto = Timestamp()
  start_dt_proto.FromDatetime(args.start_datetime)

  final_json = {
    # metrics values are numerical
    "metrics" : {
      # Traffic
      "num_prompts_attempted": benchmark_result['num_prompts_attempted'],
      "num_prompts_succeeded": benchmark_result['num_prompts_succeeded'],
      "request_rate": args.request_rate,
      "queries_per_second": benchmark_result['queries_per_second'],
      'server_metrics': {
        **server_metrics
      },
      **benchmark_result,
      **errors,
    },
    # dimensions values are strings
    "dimensions": {
      "date": args.start_datetime.strftime('%Y%m%d-%H%M%S'),
      "backend": args.backend,
      "model_id": model,
      "tokenizer_id": args.tokenizer,
      **(json.loads(args.additional_metadata_metrics_to_save) if args.additional_metadata_metrics_to_save else {})
    },
    "config": {
      "model": model,
      "num_models": len(args.models.split(',')),
      "model_server": args.backend,
      "start_time": {
        "seconds" : start_dt_proto.seconds,
        "nanos" : start_dt_proto.nanos
      }
    },
    "summary_stats": {
      "stats": [{
        "request_rate": args.request_rate,
        "request_latency": {
          "mean": benchmark_result["avg_latency_ms"],
          "median": benchmark_result["median_latency_ms"],
          "sd": benchmark_result["sd_latency_ms"],
          "min": benchmark_result["min_latency_ms"],
          "max": benchmark_result["max_latency_ms"],
          "p90": benchmark_result["p90_latency_ms"],
          "p99": benchmark_result["p99_latency_ms"],
        },
        "throughput": {
          "mean": benchmark_result['throughput']
        },
        "input_length": {
          "mean": benchmark_result["avg_input_len"],
          "median": benchmark_result["median_input_len"],
          "sd": benchmark_result["sd_input_len"],
          "min": benchmark_result["min_input_len"],
          "max": benchmark_result["max_input_len"],
          "p90": benchmark_result["p90_input_len"],
          "p99": benchmark_result["p99_input_len"],
        },
        "output_length": {
          "mean": benchmark_result["avg_output_len"],
          "median": benchmark_result["median_output_len"],
          "sd": benchmark_result["sd_output_len"],
          "min": benchmark_result["min_output_len"],
          "max": benchmark_result["max_output_len"],
          "p90": benchmark_result["p90_output_len"],
          "p99": benchmark_result["p99_output_len"],
        },
        "tpot": {
          "mean": benchmark_result["avg_normalized_time_per_output_token_ms"],
          "median": benchmark_result["median_normalized_time_per_output_token_ms"],
          "sd": benchmark_result["sd_normalized_time_per_output_token_ms"],
          "min": benchmark_result["min_normalized_time_per_output_token_ms"],
          "max": benchmark_result["max_normalized_time_per_output_token_ms"],
          "p90": benchmark_result["p90_normalized_time_per_output_token_ms"],
          "p99": benchmark_result["p99_normalized_time_per_output_token_ms"],
        },
        "model_server_metrics" : [{"Name": name, **metrics} for name, metrics in server_metrics.items()]
      }]
    }
  }
  
  # Save to file
  model_without_slash = model.replace("/","-")
  file_name = (
      f"{args.file_prefix}-{args.backend}-{args.request_rate}qps-{args.start_datetime.strftime('%Y%m%d-%H%M%S')}-{model_without_slash}.json"
  )
  with open(file_name, "w", encoding="utf-8") as outfile:
    json.dump(final_json, outfile)
  if gcs_bucket is not None:
    try:
      gcs_bucket.blob(f"{args.output_bucket_filepath}/{file_name}").upload_from_filename(file_name)
      print(f"File {file_name} uploaded to gs://{args.output_bucket}/{args.output_bucket_filepath}")
    except google.cloud.exceptions.NotFound:
      print(f"GS Bucket (gs://{args.output_bucket}) does not exist")
      
  if args.spanner_instance_id and args.spanner_database_id and spanner_upload:
    # Upload to Spanner
    try:
      upload_to_spanner_batch_with_retry(
          args.spanner_instance_id, args.spanner_database_id, [file_name],
          args.output_bucket, args.file_prefix)
      print(f"File {file_name} uploaded to Spanner")
    except Exception as e:
      print(f"Failed to upload {file_name} to Spanner: {e}")


def metrics_to_scrape(backend: str) -> List[str]:
  # Each key in the map is a metric, it has a corresponding 'stats' object
  # It must be populated on the outputs 'metrics' field as 'key':'stats'
  # If a value is specified for a given key, it will be populated on the outputs `summary_stats.stats` field as 'value':'stats' as well.
  if backend == "vllm":
    return [
      "vllm:cpu_cache_usage_perc",
      "vllm:gpu_cache_usage_perc",

      "vllm:num_requests_waiting",
      "vllm:num_requests_running",
      "vllm:num_requests_swapped",

      "vllm:time_to_first_token_seconds",
      "vllm:time_per_output_token_seconds",
      "vllm:e2e_request_latency_seconds",

      "vllm:request_prefill_time_seconds",
      "vllm:request_queue_time_seconds",
      "vllm:request_decode_time_seconds",
      "vllm:request_inference_time_seconds",
      "vllm:time_in_queue_requests",

      "vllm:request_prompt_tokens",
      "vllm:request_generation_tokens",
      "vllm:iteration_tokens_total",
      "vllm:prompt_tokens_total",
      "vllm:generation_tokens_total",
      "vllm:request_success_total",
      "vllm:num_preemptions_total",

      "vllm:cpu_prefix_cache_hit_rate",
      "vllm:gpu_prefix_cache_hit_rate",

      "vllm:avg_generation_throughput_toks_per_s",
      "vllm:avg_prompt_throughput_toks_per_s",
    ]
  elif backend == "jetstream":
    return [
      "jetstream_slots_used_percentage",
      "jetstream_prefill_backlog_size",
    ]
  else:
    return []

def print_metrics(metrics: List[str], duration_sec: float, namespace: str, job: str):
  # Creates a credentials object from the default service account file
  # Assumes that script has appropriate default credentials set up, ref:
  # https://googleapis.dev/python/google-auth/latest/user-guide.html#application-default-credentials
  credentials, project_id = google.auth.default()
  # Prepare an authentication request - helps format the request auth token
  auth_req = google.auth.transport.requests.Request()

  server_metrics = {}

  # Request refresh tokens
  credentials.refresh(auth_req)
  url='https://monitoring.googleapis.com/v1/projects/%s/location/global/prometheus/api/v1/metadata' % (project_id)
  headers_api = {'Authorization': 'Bearer ' + credentials.token}
  request_post = requests.get(url=url, headers=headers_api)
  all_metrics_metadata = request_post.json()
  if request_post.ok is not True:
    print("HTTP Error: %s" % (all_metrics_metadata))
    return server_metrics
  if all_metrics_metadata["status"] != "success":
    print("Metadata error response: %s" % all_metrics_metadata["error"])
    return server_metrics

  for metric in metrics:
    # Find metric type
    if metric not in all_metrics_metadata['data']:
      logger.debug(f"No metric found for {metric}")
      continue
    metric_type = all_metrics_metadata['data'][metric]
    metric_type = metric_type[0]['type']

    metric_results = {}
    # Queries scrape all metrics collected from the last $DURATION seconds from the backend's related
    # podmonitoring spec assumed to be named "$BACKEND-podmonitoring"

    filters = ""
    if job != "":
        filters += f'job="{job}"'
    if namespace != "":
        if filters != "":
            filters += ","
        filters += f'namespace="{namespace}"'
    if filters != "":
        filters = f"{{{filters}}}"

    queries = {
        "gauge": {
            "Mean": f"avg_over_time({metric}{filters}[{duration_sec:.0f}s])",
            "Median": f"quantile_over_time(0.5, {metric}{filters}[{duration_sec:.0f}s])",
            "Sd": f"stddev_over_time({metric}{filters}[{duration_sec:.0f}s])",
            "Min": f"min_over_time({metric}{filters}[{duration_sec:.0f}s])",
            "Max": f"max_over_time({metric}{filters}[{duration_sec:.0f}s])",
            "P90": f"quantile_over_time(0.9, {metric}{filters}[{duration_sec:.0f}s])",
            "P95": f"quantile_over_time(0.95, {metric}{filters}[{duration_sec:.0f}s])",
            "P99": f"quantile_over_time(0.99, {metric}{filters}[{duration_sec:.0f}s])",
        },
        "histogram": {
            "Mean": f"sum(rate({metric}_sum{filters}[{duration_sec:.0f}s])) / sum(rate({metric}_count{filters}[{duration_sec:.0f}s]))",
            "Median": f"histogram_quantile(0.5, sum(rate({metric}_bucket{filters}[{duration_sec:.0f}s])) by (le))",
            "Min": f"histogram_quantile(0, sum(rate({metric}_bucket{filters}[{duration_sec:.0f}s])) by (le))",
            "Max": f"histogram_quantile(1, sum(rate({metric}_bucket{filters}[{duration_sec:.0f}s])) by (le))",
            "P90": f"histogram_quantile(0.9, sum(rate({metric}_bucket{filters}[{duration_sec:.0f}s])) by (le))",
            "P95": f"histogram_quantile(0.95, sum(rate({metric}_bucket{filters}[{duration_sec:.0f}s])) by (le))",
            "P99": f"histogram_quantile(0.99, sum(rate({metric}_bucket{filters}[{duration_sec:.0f}s])) by (le))",
        },
        "counter": {
            "Sum": f"sum_over_time({metric}{filters}[{duration_sec:.0f}s])",
            "Rate": f"rate({metric}{filters}[{duration_sec:.0f}s])",
            "Increase": f"increase({metric}{filters}[{duration_sec:.0f}s])",
            "Mean": f"avg_over_time(rate({metric}{filters}[{duration_sec:.0f}s])[{duration_sec:.0f}s:{duration_sec:.0f}s])",
            "Max": f"max_over_time(rate({metric}{filters}[{duration_sec:.0f}s])[{duration_sec:.0f}s:{duration_sec:.0f}s])",
            "Min": f"min_over_time(rate({metric}{filters}[{duration_sec:.0f}s])[{duration_sec:.0f}s:{duration_sec:.0f}s])",
            "P90": f"quantile_over_time(0.9, rate({metric}{filters}[{duration_sec:.0f}s])[{duration_sec:.0f}s:{duration_sec:.0f}s])",
            "P95": f"quantile_over_time(0.95, rate({metric}{filters}[{duration_sec:.0f}s])[{duration_sec:.0f}s:{duration_sec:.0f}s])",
            "P99": f"quantile_over_time(0.99, rate({metric}{filters}[{duration_sec:.0f}s])[{duration_sec:.0f}s:{duration_sec:.0f}s])",
        },
    }

    for query_name, query in queries[metric_type].items():
      # Configure respective query
      url='https://monitoring.googleapis.com/v1/projects/%s/location/global/prometheus/api/v1/query' % (project_id)
      headers_api = {'Authorization': 'Bearer ' + credentials.token}
      params = {'query': query}
      logger.debug(f"Finding {query_name} {metric} with the following query: {query}")
      request_post = requests.get(url=url, headers=headers_api, params=params)
      response = request_post.json()

      logger.debug(f"Got response from metrics server: {response}")

      # handle response
      if request_post.ok:
        if response["status"] == "success" and response["data"] and response["data"]["result"]:
          r = response["data"]["result"]
          if not r:
            logger.debug(f"Failed to get result for {query_name}")
            continue
          v = r[0].get("value", None)
          if not v:
            logger.debug(f"Failed to get value for result: {r}")
            continue
          metric_results[query_name] = float(v[1])
          logger.debug("%s: %s" % (query_name, v[1]))
        else:
          logger.debug("Cloud Monitoring PromQL Error: %s" % (response))
          continue
      else:
        logger.debug("HTTP Error: %s" % (response))
        continue
    server_metrics[metric] = metric_results
  
  return server_metrics

def get_stats_for_set(name, description, points):
  avg = np.mean(points) if points else 0
  median = np.median(points) if points else 0
  sd = np.std(points) if points else 0
  min_val = np.min(points) if points else 0
  max_val = np.max(points) if points else 0
  p90 = np.percentile(points, 90) if points else 0
  p99 = np.percentile(points, 99) if points else 0

  print(f"Average {description}: {avg:.2f}")

  return {
    f'avg_{name}':  avg,
    f'median_{name}': median,
    f'sd_{name}': sd,
    f'min_{name}': min_val,
    f'max_{name}': max_val,
    f'p90_{name}': p90,
    f'p99_{name}': p99,
  }

async def print_and_save_result(
    args: argparse.Namespace, 
    benchmark_duration_sec, 
    total_requests, 
    model, 
    request_latencies, 
    server_ttfts, 
    client_ttfts,
    predicted_ttfts, 
    server_tpots, 
    client_tpots,
    predicted_tpots,
    server_predicted_ttft_pairs,  # New parameter
    server_predicted_tpot_pairs,  # New parameter
    errors, 
    slo_met_count=None,
    spanner_upload=False, 
    server_metrics_scrape=False
):
    benchmark_result = {}

    print(f"====Result for Model: {model}====")
    print(f"Errors: {errors}")
    print(f"Total time (seconds): {benchmark_duration_sec:.2f} s")
    print(f"Successful/total requests: {len(request_latencies)}/{total_requests}")
    print(f"Requests/sec: {total_requests / benchmark_duration_sec:.2f}")
    print(f"SLO met count: {slo_met_count if slo_met_count is not None else 'N/A'}")
    counter = await AsyncRequestCounter()
    queries_per_second = await counter.get_qps()
    print(f"Queries/sec: {queries_per_second:.2f}")
    benchmark_result['queries_per_second'] = queries_per_second
    benchmark_result["num_prompts_attempted"] = total_requests
    benchmark_result["num_prompts_succeeded"] = len(request_latencies)
    benchmark_result['throughput_rps'] = (args.num_prompts / benchmark_duration_sec)
    benchmark_result['benchmark_time'] = benchmark_duration_sec
    benchmark_result['slo_met_count'] = slo_met_count if slo_met_count is not None else benchmark_result["num_prompts_succeeded"]
    benchmark_result['slo_met_perc'] = slo_met_count / benchmark_result["num_prompts_attempted"] * 100 if benchmark_result["num_prompts_attempted"] > 0 and slo_met_count is not None else 0

    total_output_tokens = np.sum([output_len for _, output_len, _ in
                                  request_latencies])
    output_tokens_per_second = total_output_tokens / benchmark_duration_sec
    benchmark_result['throughput'] = output_tokens_per_second

    print(f"Output_tokens/sec: {output_tokens_per_second:.2f}")
    benchmark_result['total_output_token'] = int(total_output_tokens)

    total_input_tokens = np.sum([prompt_len for prompt_len, _, _ in
                                 request_latencies])
    input_tokens_per_sec = total_input_tokens / benchmark_duration_sec
    print(f"Input_tokens/sec: {input_tokens_per_sec:.2f}")
    benchmark_result['total_input_tokens'] = int(total_input_tokens)
    benchmark_result['input_tokens_per_sec'] = input_tokens_per_sec

    total_tokens = total_input_tokens + total_output_tokens
    tokens_per_sec = total_tokens / benchmark_duration_sec
    print(f"Tokens/sec: {tokens_per_sec:.2f}")
    benchmark_result['total_tokens'] = int(total_tokens)
    benchmark_result['tokens_per_sec'] = tokens_per_sec
    
    # Process server and client TTFT metrics
    server_ttft_stats = {}
    client_ttft_stats = {}
    combined_ttft_stats = {}
    
    # Process server and client TPOT metrics
    server_tpot_stats = {}
    client_tpot_stats = {}
    combined_tpot_stats = {}
    
    # Process predicted metrics
    predicted_ttft_stats = {}
    predicted_tpot_stats = {}
    
    # NEW: Process prediction error metrics
    ttft_prediction_error_stats = {}
    tpot_prediction_error_stats = {}
    
    ttft_per_input_token_stats = {}
    
    if args.stream_request:
        # Server TTFT stats
        if server_ttfts:
            server_ttft_stats = get_stats_for_set("server_TTFT_ms", "Server Time to First Token (ms)", server_ttfts)
            print("Server TTFT metrics:")
            for key, value in server_ttft_stats.items():
                print(f"  {key}: {value:.2f}")
        
        # Client TTFT stats  
        if client_ttfts:
            client_ttft_stats = get_stats_for_set("client_TTFT_ms", "Client Time to First Token (ms)", client_ttfts)
            print("Client TTFT metrics:")
            for key, value in client_ttft_stats.items():
                print(f"  {key}: {value:.2f}")
        
        # Combined TTFT (prefer server when available, fall back to client)
        combined_ttfts = []
        for i in range(len(request_latencies)):
            if i < len(server_ttfts) and server_ttfts[i] > 0:
                combined_ttfts.append(server_ttfts[i])
            elif i < len(client_ttfts) and client_ttfts[i] > 0:
                combined_ttfts.append(client_ttfts[i])
        
        if combined_ttfts:
            combined_ttft_stats = get_stats_for_set("TTFT_ms", "Combined Time to First Token (ms)", combined_ttfts)
        
        # Server TPOT stats
        if server_tpots:
            server_tpot_stats = get_stats_for_set("server_TPOT_ms", "Server Time Per Output Token (ms)", server_tpots)
            print("Server TPOT metrics:")
            for key, value in server_tpot_stats.items():
                print(f"  {key}: {value:.2f}")
        
        # Client TPOT stats
        if client_tpots:
            client_tpot_stats = get_stats_for_set("client_TPOT_ms", "Client Time Per Output Token (ms)", client_tpots)
            print("Client TPOT metrics:")
            for key, value in client_tpot_stats.items():
                print(f"  {key}: {value:.2f}")
        
        # Combined TPOT (prefer server when available, fall back to client)
        combined_tpots = []
        for i in range(len(request_latencies)):
            if i < len(server_tpots) and server_tpots[i] > 0:
                combined_tpots.append(server_tpots[i])
            elif i < len(client_tpots) and client_tpots[i] > 0:
                combined_tpots.append(client_tpots[i])
        
        if combined_tpots:
            combined_tpot_stats = get_stats_for_set("TPOT_ms", "Combined Time Per Output Token (ms)", combined_tpots)
        
        # Predicted metrics
        if predicted_ttfts:
            predicted_ttft_stats = get_stats_for_set("predicted_TTFT_ms", "Predicted Time to First Token (ms)", predicted_ttfts)
        if predicted_tpots:
            predicted_tpot_stats = get_stats_for_set("predicted_TPOT_ms", "Predicted Time Per Output Token (ms)", predicted_tpots)
        
        # NEW: Calculate prediction error metrics
        if server_predicted_ttft_pairs:
            server_ttfts_for_error = [pair[0] for pair in server_predicted_ttft_pairs]
            predicted_ttfts_for_error = [pair[1] for pair in server_predicted_ttft_pairs]
            ttft_prediction_error_stats = get_prediction_error_stats("TTFT", server_ttfts_for_error, predicted_ttfts_for_error)
        
        if server_predicted_tpot_pairs:
            server_tpots_for_error = [pair[0] for pair in server_predicted_tpot_pairs]
            predicted_tpots_for_error = [pair[1] for pair in server_predicted_tpot_pairs]
            tpot_prediction_error_stats = get_prediction_error_stats("TPOT", server_tpots_for_error, predicted_tpots_for_error)
        
        # TTFT per input token calculation
        ttft_per_input_token = []
        for i, (prompt_len, _, _) in enumerate(request_latencies):
            if prompt_len > 0:
                if i < len(combined_ttfts):
                    ttft_per_input_token.append(combined_ttfts[i] / prompt_len)
                elif i < len(server_ttfts) and server_ttfts[i] > 0:
                    ttft_per_input_token.append(server_ttfts[i] / prompt_len)
                elif i < len(client_ttfts) and client_ttfts[i] > 0:
                    ttft_per_input_token.append(client_ttfts[i] / prompt_len)
        
        if ttft_per_input_token:
            ttft_per_input_token_stats = get_stats_for_set("TTFT_per_input_token_ms", "Time to First Token per Input Token (ms/token)", ttft_per_input_token)

    if args.machine_cost:
        print(
            "Cost $/1k tokens:"
            f" {args.machine_cost * 1000 / output_tokens_per_second}"
        )

    benchmark_result = {
        **benchmark_result,
        **(get_stats_for_set("per_token_latency_ms", "milliseconds/token (includes waiting time on server)", [
            latency / (prompt_len + output_len)
            for prompt_len, output_len, latency in request_latencies
        ])),
        **combined_ttft_stats,
        **server_ttft_stats,
        **client_ttft_stats,
        **ttft_per_input_token_stats,
        **combined_tpot_stats,
        **server_tpot_stats,
        **client_tpot_stats,
        **predicted_ttft_stats,
        **predicted_tpot_stats,
        **ttft_prediction_error_stats,  # NEW: Add prediction error stats
        **tpot_prediction_error_stats,  # NEW: Add prediction error stats
        # NOTE: The latency below includes requests awaiting time on server side.
        # It's not comparable with the model inference latency for batch size 1.
        **(get_stats_for_set("latency_ms", "milliseconds/request (includes waiting time on server)" ,[latency for _, _, latency in request_latencies])),
        **(get_stats_for_set("normalized_time_per_output_token_ms", "milliseconds/output_token (includes waiting time on server)", [latency / output_len for _, output_len, latency in request_latencies])),
        **(get_stats_for_set("input_len", "input length", [float(prompt_len) for prompt_len, _, _ in request_latencies])),
        **(get_stats_for_set("output_len", "output length", [float(output_len) for _, output_len, _ in request_latencies]))
    }

    server_metrics = {}
    if args.scrape_server_metrics and server_metrics_scrape:
        server_metrics = print_metrics(metrics_to_scrape(args.backend), benchmark_duration_sec, args.pm_namespace, args.pm_job)
    if args.save_json_results:
        save_json_results(args, benchmark_result, server_metrics, model, errors, spanner_upload)

async def main(args: argparse.Namespace):
  print(args)
  models = args.models.split(',')
  print(f"Models to benchmark: {models}")
  if args.traffic_split:
    print(f"Traffic split: {args.traffic_split}")
  else:
    print("No traffic split specified. Defaulting to uniform traffic split.")
  random.seed(args.seed)
  np.random.seed(args.seed)
  endpoint = (
    "v1/completions"
    if args.backend == "vllm"
    else args.endpoint
)
  
  # Create GCS client before benchmarking
  # Should fail fast if client is misconfigured or missing permissions
  if args.output_bucket is not None:
    global gcs_client
    gcs_client = storage.Client()
    global gcs_bucket
    gcs_bucket = gcs_client.bucket(args.output_bucket)

    if args.output_bucket_filepath:
      blob = gcs_bucket.blob(args.output_bucket_filepath)
      if not blob.exists():
        blob.upload_from_string('')

  print(f"Starting Prometheus Server on port {args.prometheus_port}")
  start_http_server(args.prometheus_port)

  api_url = f"http://{args.host}:{args.port}/{endpoint}"
  tokenizer = AutoTokenizer.from_pretrained(
      args.tokenizer, trust_remote_code=args.trust_remote_code
  )

  benchmark_start_time = time.time()
  args.start_datetime = datetime.fromtimestamp(benchmark_start_time)
  
  await benchmark(args, api_url, tokenizer,models, args.traffic_split)
  



def parse_traffic_split(arg):
    try:
        return [float(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Traffic split must be a comma-separated list of floats, e.g. '0.9,0.1'"
        )

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Benchmark the online serving throughput."
  )
  parser.add_argument(
      "--backend",
      type=str,
      default="vllm",
      choices=[
          "vllm",
          "tgi",
          "naive_transformers",
          "tensorrt_llm_triton",
          "sax",
          "jetstream"
      ],
  )
  parser.add_argument(
      "--sax_model",
      type=str,
      default="",
      help="Model name to send request to at API server for SAX model server.",
  )
  parser.add_argument("--file-prefix", type=str, default="benchmark")
  parser.add_argument("--endpoint", type=str, default="generate")
  parser.add_argument("--host", type=str, default="localhost")
  parser.add_argument("--port", type=int, default=7080)
  parser.add_argument("--dataset", type=str, help="Path to the dataset.")
  parser.add_argument(
    "--models",
    type=str,
    help="Comma separated list of models to benchmark.",
  )
  parser.add_argument(
    "--traffic-split",
    type=parse_traffic_split,
    default=None,
    help="Comma-separated list of traffic split proportions for the models, e.g. '0.9,0.1'. Sum must equal 1.0."
)
  parser.add_argument(
    "--stream-request", 
    action="store_true",
    help="Whether to stream the request. Needed for TTFT metric",
  )
  parser.add_argument(
    "--request-timeout", 
    type=float,
    default=(3.0 * 60.0 * 60.0),
    help="Individual request timeout",
  )
  parser.add_argument(
      "--tokenizer",
      type=str,
      required=True,
      help="Name or path of the tokenizer.",
  )
  parser.add_argument(
      "--best-of",
      type=int,
      default=1,
      help="Generates `best_of` sequences per prompt and returns the best one.",
  )
  parser.add_argument("--use-beam-search", action="store_true")
  parser.add_argument(
      "--num-prompts",
      type=int,
      default=1000,
      help="Number of prompts to process.",
  )
  parser.add_argument(
      "--max-input-length",
      type=int,
      default=1024,
      help=(
          "Maximum number of input tokens for filtering the benchmark dataset."
      ),
  )
  parser.add_argument(
      "--max-output-length",
      type=int,
      default=1024,
      help=(
          "Maximum number of output tokens for filtering the benchmark dataset."
      ),
  )
  parser.add_argument(
      "--min-input-length",
      type=int,
      default=4,
      help=(
          "Minimum number of input tokens for filtering the benchmark dataset."
      ),
  )
  parser.add_argument(
      "--min-output-length",
      type=int,
      default=4,
      help=(
          "Minimum number of output tokens for filtering the benchmark dataset."
      ),
  )
  parser.add_argument(
    "--ignore-eos",
    action="store_true",
    default=True,
    help=(
        "If set and model server is vllm, the generation process will ignore the end-of-sequence (EOS) token, "
        "allowing output to continue until reaching --max-output-length or another stopping condition."
    ),
  )
  parser.add_argument(
      "--top-k",
      type=int,
      default=32000,
      help=(
          "Number of candidate tokens that are considered at each step of the"
          " generation process. 32000 is the vocab_size of Open-LLaMA and"
          " LLaMA2 models."
      ),
  )
  parser.add_argument(
      "--request-rate",
      type=float,
      default=float("inf"),
      help=(
          "Number of requests per second. If this is inf, "
          "then all the requests are sent at time 0. "
          "Otherwise, we use Poisson process to synthesize "
          "the request arrival times."
      ),
  )
  parser.add_argument("--seed", type=int, default=int(time.time()))
  parser.add_argument(
      "--trust-remote-code",
      action="store_true",
      help="trust remote code from huggingface",
  )
  parser.add_argument(
      "--machine-cost",
      type=float,
      default=None,
      help="Machine cost per hour including accelerators (if any)",
  )
  parser.add_argument(
      "--use-dummy-text",
      action="store_true",
      help=(
          "Whether to use dummy text with length defined by max_input_length"
          " and max_output_length."
      ),
  )
  parser.add_argument(
      "--save-json-results",
      action="store_true",
      help="Whether to save benchmark results to a json file.",
  )
  parser.add_argument(
    "--output-bucket",
    type=str,
    default=None,
    help=(
      "Specifies the Google Cloud Storage bucket to which JSON-format results"
      " will be uploaded. If not provided, no upload will occur."
    )
  )
  parser.add_argument(
    "--output-bucket-filepath",
    type=str,
    default=None,
    help=(
      "Specifies the destination path within the bucket provided by"
      " --output-bucket for uploading the JSON results. This argument requires"
      " --output-bucket to be set. If not specified, results will be uploaded "
      " to the root of the bucket. If the filepath doesnt exist, it will be"
      " created for you."
    )
  )
  parser.add_argument(
    "--save-aggregated-result",
    action="store_true",
    help="Whether to aggregate results of all models and save the result.",
  )
  parser.add_argument(
      "--additional-metadata-metrics-to-save",
      type=str,
      help=(
          "Additional metadata about the workload. Should be a dictionary in"
          " the form of a string."
      ),
  )
  parser.add_argument(
      "--scrape-server-metrics",
      action="store_true",
      help="Whether to scrape server metrics.",
  )
  parser.add_argument(
       "--spanner-instance-id",
       type=str,
       default=None,
       help="Spanner instance ID to upload results to.",
  )
  parser.add_argument(
       "--spanner-database-id",
       type=str,
       default=None,
       help="Spanner database ID to upload results to.",
  )
  
  parser.add_argument(
     "--ttft-slo",
     type=float,
     default=None,
     help="Desired ttft SLO in milliseconds",
   )
  parser.add_argument(
     "--avg-tpot-slo",
     type=float,
     default=None,
     help="Desired average tpot SLO in milliseconds",
   )
  parser.add_argument(
    "--enable-slo-based-routing",
    action="store_true",
    default=False,
     help="Enable SLO-based routing. If set, the benchmark will route requests based on the SLOs provided for ttft and tpot.",
   )
  
  parser.add_argument(
     "--prometheus-port",
     type=int,
     default=9090,
     help="Port for Prometheus metrics",
   )

  
  parser.add_argument("--pm-namespace", type=str, default="default", help="namespace of the pod monitoring object, ignored if scrape-server-metrics is false")
  parser.add_argument("--pm-job", type=str, default="vllm-podmonitoring", help="name of the pod monitoring object, ignored if scrape-server-metrics is false")
  parser.add_argument("--tcp-conn-limit", type=int, default=100, help="Max number of tcp connections allowed per aiohttp ClientSession")
  cmd_args = parser.parse_args()
  
  level = logging.INFO
  logger = logging.getLogger(__name__)
  logger.setLevel(level)
  handler = logging.StreamHandler()  # This sends output to the console
  handler.setLevel(level) # Set handler level
  logger.addHandler(handler)
  
  asyncio.run(main(cmd_args))