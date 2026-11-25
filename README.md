# spark-agent

A minimal Go CLI harness for running a tool-calling coding agent against a local OpenAI-compatible vLLM endpoint (e.g., NVIDIA DGX Spark). The agent operates inside a sandboxed repository root, with filesystem operations confined via `os.OpenRoot` and shell commands gated behind interactive user confirmation.

## Architecture

```
spark-agent/
├── cmd/spark-agent/main.go   # CLI entry: flag parsing, REPL loop, signal handling
├── internal/
│   ├── agent/loop.go         # Turn state machine: prompt -> stream -> tool calls -> prune
│   ├── cli/                  # Multiline REPL prompt reader and input history
│   ├── llm/client.go         # OpenAI-compatible HTTP/SSE client (/v1/models, /v1/chat/completions)
│   ├── tools/registry.go     # Typed tool definitions, os.Root sandboxing, batch execution
│   └── vllm/manager.go       # Model zoo coordinator: up/down/status, sleep/wake, primary switching
```

### Components

| Package | Role |
| --------- | ------ |
| `cmd/spark-agent` | Parses flags, displays active model and context window info, starts the REPL, and manages per-turn timeouts and Ctrl+C cancellation |
| `internal/agent` | `Agent` struct holds conversation history, streams chat completions, prunes older turns near the context limit, and loops on tool calls (max 15 steps per turn) |
| `internal/cli` | Interactive terminal prompt reader with multiline input and command history |
| `internal/llm` | HTTP/SSE client for OpenAI-compatible endpoints; queries active model ID and `max_model_len` and streams chat completions with token usage tracking |
| `internal/tools` | `Registry` manages tool definitions, enforces filesystem sandboxing, and executes tool calls in batch |
| `internal/vllm` | `Manager` coordinates the vLLM model zoo: starts/stops containers, sleeps/wakes models, and enforces unified memory by sleeping all other primary models when switching |

## Features

- **Streaming responses & tool-calling loop** - streams assistant tokens live via SSE (`stream_options.include_usage`), executes returned tool calls, feeds results back, and repeats until the model responds without tool calls.
- **Automatic context window budgeting** - queries `max_model_len` from vLLM `/v1/models` (or accepts a `-max-context` override) and automatically prunes the oldest user turns when token usage reaches 80% of the context limit.
- **Filesystem sandboxing** - all file operations (`list_files`, `read_file`, `write_file`, `edit_file`) are confined to the repository root via Go 1.22+'s `os.OpenRoot`. Paths are also cleaned with `filepath.Clean` as defense-in-depth.
- **Read-only / mutating separation** - read-only tools run concurrently (up to 4 workers); mutating tools run sequentially after all reads complete.
- **Bash with human-in-the-loop** - `run_bash` sets the working directory to the repo root but requires explicit `y` confirmation from the user before executing any command.
- **Per-turn timeout & cancellation** - each turn has a configurable timeout (default 5 min) and responds to `Ctrl+C` mid-turn.
- **Model zoo integration** - manage multiple vLLM models with unified memory enforcement. Switch between models at runtime via `/model` REPL commands or the `-model` flag.

## Model Zoo

The model zoo coordinates multiple vLLM containers, each serving a different model. When switching to a primary model, all other primary models (including HighMemory ones) are automatically put to sleep to free GPU memory.

### Docker Compose

Models are defined in `deploy/vllm/docker-compose.yml`:

| Service | Port | Model |
|---------|------|-------|
| sidecar-7b | 8001 | Qwen/Qwen2.5-Coder-7B-Instruct |
| gemma4 | 8002 | google/gemma-4-26B-A4B |
| qwen3-coder | 8003 | Qwen/Qwen3-Coder-Next-FP8 |
| qwen36-35b | 8004 | Qwen/Qwen3.6-35B-A3B |

### Shell Script

```bash
# Start all models (each starts sleeping)
./deploy/vllm/zoo-ctl.sh up

# Stop all models
./deploy/vllm/zoo-ctl.sh down

# Check status of all models
./deploy/vllm/zoo-ctl.sh status
```

### Go Manager API

```go
mgr := vllm.NewManager("deploy/vllm/docker-compose.yml")

// Ensure a model is awake and get its base URL
baseURL, modelID, err := mgr.EnsureReady(ctx, "sidecar")

// Sleep all awake models
mgr.SleepAll()

// Stop all containers
mgr.DownAll()

// Get status of all models
statuses := mgr.Status()
```

## Tools

| Tool | Read-Only | Description |
| ------ | ----------- | ------------- |
| `list_files` | yes | Lists relative file paths in the repository (skips `.git`, `node_modules`, `target`) |
| `read_file` | yes | Reads the full contents of a file |
| `write_file` | no | Creates or overwrites a file |
| `edit_file` | no | Replaces an exact, unique text snippet in an existing file |
| `run_bash` | no | Runs a shell command (requires interactive `y` confirmation) |

## Usage

```bash
# Build
go build -o spark-agent ./cmd/spark-agent

# Run (defaults to local vLLM at http://localhost:8000/v1, current directory as sandbox)
./spark-agent

# Custom base URL and sandbox directory
./spark-agent -url http://192.168.1.100:8000/v1 -root /path/to/repo

# Custom per-turn timeout and context window override
./spark-agent -timeout 10m -max-context 65536

# Use a specific model from the zoo (auto-starts and switches)
./spark-agent -model sidecar

# Use a custom docker-compose file
./spark-agent -model qwen3-coder -compose /path/to/docker-compose.yml
```

### REPL Commands

| Input | Action |
|-------|--------|
| Any text | Send as a prompt to the agent |
| `/exit` or `/quit` | Terminate the session |
| `/model` or `/model up` | Ensure current model is awake |
| `/model <name>` | Switch to a different model (e.g., `/model gemma4`) |
| `/model down` | Stop all model containers |
| `/model status` | Show status of all models |
| `/model list` | List all available models |
| `Ctrl+C` | Cancel the current turn mid-execution |

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-url` | `http://localhost:8000/v1` | OpenAI-compatible vLLM base URL |
| `-model` | (empty) | Model name or alias from the zoo (e.g. sidecar, qwen3-coder, gemma4) |
| `-compose` | (auto-detect) | Path to docker-compose.yml for the model zoo |
| `-root` | `.` | Repository root directory for filesystem sandboxing |
| `-timeout` | `5m` | Per-turn timeout duration |
| `-max-context` | `0` | Override context window size in tokens (`0` = auto-detect from vLLM `max_model_len`, fallback `32768`) |

## Requirements

- Go 1.22+ (for `os.OpenRoot` filesystem sandboxing)
- Docker and Docker Compose (for model zoo management)
- A running OpenAI-compatible inference server (e.g., vLLM on NVIDIA DGX Spark)
