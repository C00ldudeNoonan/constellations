# vLLM provider

The built-in `vllm` inference provider targets a local or remote
[vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/).
It uses Chat Completions and requests
[JSON-schema structured output](https://docs.vllm.ai/en/latest/features/structured_outputs/)
for document extraction. No OpenAI client dependency is required.

## Start a local server

Choose an instruction-tuned model with a chat template. The API model in the
dbt-ml profile must match either the model passed to `vllm serve` or an explicit
`--served-model-name` alias.

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --served-model-name invoice-extractor \
  --api-key local-vllm-token \
  --generation-config vllm
```

`--generation-config vllm` prevents a model repository's
`generation_config.json` from replacing request sampling defaults. The server
listens on port 8000 by default, so configure the OpenAI-compatible `/v1` base:

```yaml
my_project:
  target: dev
  outputs:
    dev:
      warehouse:
        type: duckdb
        path: ./target/dbt_ml.duckdb
        schema: documents
      llm:
        provider: vllm
        model: invoice-extractor
        base_url: http://127.0.0.1:8000/v1
        api_key_env: VLLM_API_KEY
        timeout_seconds: 120
        cache_path: ./target/llm_cache.duckdb
```

Set `VLLM_API_KEY=local-vllm-token` in the process running dbt-ml. If the vLLM
server was deliberately started without `--api-key`, omit `api_key_env`; local
unauthenticated endpoints are supported.

## Model configuration

Models remain provider-neutral. The provider, endpoint, and credential
reference are operator-owned profile settings:

```yaml
version: 2
models:
  - name: extracted_invoices
    source: ref('invoice_text')
    extraction:
      backend: llm
      options:
        fields:
          - name: invoice_id
            type: string
          - name: total
            type: number
        temperature: 0
        max_tokens: 2048
        max_concurrent: 8
```

dbt-ml does not submit vLLM's native batch endpoint, so `batch: true` is not
available for this provider. Use normal parallel extraction and tune
`max_concurrent` to give the server enough simultaneous requests to schedule
efficiently without overwhelming its queue. Start conservatively, then measure
throughput and tail latency for the deployed model and hardware.

## Remote, Docker, and Kubernetes endpoints

`base_url` may point to any reachable HTTP(S) OpenAI-compatible `/v1` base, for
example `https://inference.example.com/v1` or an internal Kubernetes service
such as `http://vllm.default.svc.cluster.local:8000/v1`. Use HTTPS and bearer
authentication whenever traffic crosses a trusted local network boundary.
Keep service credentials in the environment or your platform's secret store;
never place tokens in `base_url`.

The request timeout applies to each HTTP attempt and accepts values from 0.1 to
3600 seconds. `max_retries` controls retries for timeouts, network failures,
rate limits, and transient HTTP statuses. Remote deployments generally need a
longer timeout than local inference, especially for large documents or a cold
model.

The normalized endpoint is part of response-cache and incremental-state
identity. Switching a profile from one deployment to another cannot silently
reuse the first deployment's output. Manifests record only a one-way endpoint
fingerprint, not the infrastructure URL.
