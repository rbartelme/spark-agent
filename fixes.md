# Task: Add Versioned vLLM Model Zoo & Hot-Swap Support to `spark-agent`

We are upgrading `github.com/rbartelme/spark-agent` to manage and hot-swap a local vLLM model zoo on a unified-memory NVIDIA DGX Spark using **vLLM Level 2 Sleep Mode**. Because all models share unified memory, only one primary model may be awake at a time, and the 7B sidecar must sleep whenever the high-memory `Qwen3-Coder-Next-FP8` (85% memory utilization) model is active.

Inspect the repository first (`go.mod`, `cmd/spark-agent/main.go`, `internal/llm/client.go`, `internal/agent/loop.go`, and `internal/cli/`), then implement the following changes:

---

## 1. Create `deploy/vllm/docker-compose.yml`
Create `deploy/vllm/docker-compose.yml` with each model exposed on its own dedicated localhost port, persistent `~/.cache/vllm` and `~/.cache/triton` volume mounts, `VLLM_SERVER_DEV_MODE=1`, `--enable-sleep-mode`, and `--compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8]}'`:

```yaml
x-vllm-base: &vllm-base
  image: nvcr.io/nvidia/vllm:26.10-py3
  ipc: host
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
  volumes:
    - ~/models:/models:ro
    - ~/.cache/vllm:/root/.cache/vllm
    - ~/.cache/triton:/root/.cache/triton
  environment:
    - VLLM_NO_USAGE_STATS=1
    - HF_HUB_OFFLINE=1
    - VLLM_SERVER_DEV_MODE=1
  restart: unless-stopped

services:
  sidecar-7b:
    <<: *vllm-base
    container_name: vllm-sidecar-7b
    ports:
      - "8001:8001"
    command:
      - vllm
      - serve
      - /models/Qwen/Qwen2.5-Coder-7B-Instruct
      - --served-model-name
      - Qwen/Qwen2.5-Coder-7B-Instruct
      - --host
      - 0.0.0.0
      - --port
      - "8001"
      - --gpu-memory-utilization
      - "0.14"
      - --max-model-len
      - "16384"
      - --enable-prefix-caching
      - --enable-sleep-mode

  gemma4:
    <<: *vllm-base
    container_name: vllm-gemma4
    ports:
      - "8002:8000"
    command:
      - vllm
      - serve
      - /models/google/gemma-4-26B-A4B
      - --served-model-name
      - google/gemma-4-26B-A4B
      - --host
      - 0.0.0.0
      - --port
      - "8000"
      - --gpu-memory-utilization
      - "0.75"
      - --max-model-len
      - "32768"
      - --kv-cache-dtype
      - fp8
      - --enable-auto-tool-choice
      - --tool-call-parser
      - gemma4
      - --reasoning-parser
      - gemma4
      - --enable-prefix-caching
      - --enable-sleep-mode
      - --compilation-config
      - '{"cudagraph_capture_sizes": [1, 2, 4, 8]}'

  qwen3-coder:
    <<: *vllm-base
    container_name: vllm-qwen3-coder
    ports:
      - "8003:8000"
    command:
      - vllm
      - serve
      - /models/Qwen/Qwen3-Coder-Next-FP8
      - --served-model-name
      - Qwen/Qwen3-Coder-Next-FP8
      - --host
      - 0.0.0.0
      - --port
      - "8000"
      - --gpu-memory-utilization
      - "0.85"
      - --max-model-len
      - "32768"
      - --kv-cache-dtype
      - fp8
      - --enable-auto-tool-choice
      - --tool-call-parser
      - qwen3_xml
      - --enable-prefix-caching
      - --enable-sleep-mode
      - --compilation-config
      - '{"cudagraph_capture_sizes": [1, 2, 4, 8]}'

  qwen36-35b:
    <<: *vllm-base
    container_name: vllm-qwen36-35b
    ports:
      - "8004:8000"
    command:
      - vllm
      - serve
      - /models/Qwen/Qwen3.6-35B-A3B
      - --served-model-name
      - Qwen/Qwen3.6-35B-A3B
      - --host
      - 0.0.0.0
      - --port
      - "8000"
      - --gpu-memory-utilization
      - "0.75"
      - --max-model-len
      - "32768"
      - --kv-cache-dtype
      - fp8
      - --enable-auto-tool-choice
      - --tool-call-parser
      - qwen3_xml
      - --reasoning-parser
      - qwen3
      - --enable-prefix-caching
      - --enable-sleep-mode
      - --compilation-config
      - '{"cudagraph_capture_sizes": [1, 2, 4, 8]}'
```

---

## 2. Create `deploy/vllm/zoo-ctl.sh` (Executable)
Create `deploy/vllm/zoo-ctl.sh` (`chmod +x`) supporting `up`, `down`, and `status`:
- **`up`**: Creates `~/.cache/vllm` and `~/.cache/triton`, then sequentially starts `sidecar-7b:8001`, `gemma4:8002`, `qwen3-coder:8003`, and `qwen36-35b:8004` one at a time via `docker compose -f deploy/vllm/docker-compose.yml up -d <service>`. Wait for `GET http://localhost:<port>/health` to return 200 before calling `POST http://localhost:<port>/sleep?level=2&mode=wait` and moving to the next model so unified memory is never exceeded.
- **`down`**: Runs `docker compose -f deploy/vllm/docker-compose.yml down`.
- **`status`**: Queries `/health` and `/is_sleeping` across all four ports and prints a clean status table (`SERVICE`, `PORT`, `STATE`: `STOPPED`, `SLEEPING (L2)`, or `AWAKE`).

---

## 3. Create `internal/vllm/manager.go`
Implement package `vllm` (`://github.com`) using standard library packages (`net/http`, `sync`, `os/exec`, `encoding/json`, `context`):

1. **Model Registry & Aliases:**
   Support both full HuggingFace model names and short aliases:
   - `sidecar` / `Qwen/Qwen2.5-Coder-7B-Instruct` -> service: `sidecar-7b`, port: `8001`, `IsSidecar: true`
   - `gemma4` / `google/gemma-4-26B-A4B` -> service: `gemma4`, port: `8002`
   - `qwen3-coder` / `Qwen/Qwen3-Coder-Next-FP8` -> service: `qwen3-coder`, port: `8003`, `HighMemory: true`
   - `qwen36-35b` / `Qwen/Qwen3.6-35B-A3B` -> service: `qwen36-35b`, port: `8004`
2. **Thread-Safe `EnsureReady(ctx context.Context, modelOrAlias string) (baseURL string, modelID string, err error)`:**
   - Guard transitions with a `sync.Mutex`.
   - **Unified Memory Enforcement:**
     - When waking a primary model (`!target.IsSidecar`), sleep any other awake primary models (`POST /sleep?level=2&mode=wait`).
     - When waking `qwen3-coder` (`target.HighMemory`), also sleep `sidecar-7b` (`:8001`) if awake.
     - When waking `sidecar-7b` (`target.IsSidecar`), sleep `qwen3-coder` (`:8003`) if awake.
   - **Cold-Start Fallback:** If `GET http://localhost:<port>/is_sleeping` fails because the container isn't running, automatically invoke `docker compose -f <composeFile> up -d <service>` (locating `deploy/vllm/docker-compose.yml` via `-compose` flag, env var `SPARK_VLLM_COMPOSE`, or walking up from the executable/working directory) and poll `/health` until ready.
   - **Level 2 Wake Sequence:** If `is_sleeping` is `true`, execute in strict order:
     1. `POST /wake_up?tags=weights`
     2. `POST /collective_rpc` with body `{"method":"reload_weights"}`
     3. `POST /wake_up?tags=kv_cache`
   - Return `http://localhost:<port>/v1` and the canonical `modelID`.
3. **Lifecycle Helpers:** Expose `SleepAll`, `WarmAll`, `DownAll`, and `Status` methods on `Manager` so the CLI can manage the zoo directly in Go as well.

---

## 4. Integrate with `internal/llm/client.go`, `internal/agent/loop.go`, and `cmd/spark-agent/main.go`
1. **Dynamic Client Re-targeting (`internal/llm/client.go`):**
   - Allow updating the client's active `baseURL` (or querying `/v1/models` on the newly woken endpoint to refresh `modelID` and `max_model_len` cleanly when switching models mid-session).
2. **CLI Flags & Subcommands (`cmd/spark-agent/main.go`):**
   - Add `-model` flag (e.g., `-model qwen36-35b` or `-model Qwen/Qwen3-Coder-Next-FP8`). When `-model` is set (and `-url` is not explicitly overridden to a remote host), resolve the active endpoint through `vllm.Manager.EnsureReady`.
   - Add `-compose` flag (optional path override to `deploy/vllm/docker-compose.yml`).
   - Support subcommands `spark-agent zoo up`, `spark-agent zoo down`, and `spark-agent zoo status` before starting the interactive REPL.
3. **Interactive REPL `/model` Slash-Command (`cmd/spark-agent/main.go` & `internal/cli`):**
   - Add `/model` (lists available zoo models and highlights the active one) and `/model <name|alias>` (calls `EnsureReady`, updates the `llm.Client` endpoint and `max_model_len`, and prints the new active model and context budget without losing conversation history).
4. **Update `README.md` and Verify:**
   - Document `-model`, `/model`, and `spark-agent zoo` in `README.md`.
   - Run `go test -race ./...` and `go build -o spark-agent ./cmd/spark-agent` to verify everything compiles and passes tests cleanly.
