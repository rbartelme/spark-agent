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
│   └── tools/registry.go     # Typed tool definitions, os.Root sandboxing, batch execution
```

### Components

| Package | Role |
| --------- | ------ |
| `cmd/spark-agent` | Parses flags, displays active model and context window info, starts the REPL, and manages per-turn timeouts and Ctrl+C cancellation |
| `internal/agent` | `Agent` struct holds conversation history, streams chat completions, prunes older turns near the context limit, and loops on tool calls (max 15 steps per turn) |
| `internal/cli` | Interactive terminal prompt reader with multiline input and command history |
| `internal/llm` | HTTP/SSE client for OpenAI-compatible endpoints; queries active model ID and `max_model_len` and streams chat completions with token usage tracking |
| `internal/tools` | `Registry` manages tool definitions, enforces filesystem sandboxing, and executes tool calls in batch |

## Features

- **Streaming responses & tool-calling loop** - streams assistant tokens live via SSE (`stream_options.include_usage`), executes returned tool calls, feeds results back, and repeats until the model responds without tool calls.
- **Automatic context window budgeting** - queries `max_model_len` from vLLM `/v1/models` (or accepts a `-max-context` override) and automatically prunes the oldest user turns when token usage reaches 80% of the context limit.
- **Filesystem sandboxing** - all file operations (`list_files`, `read_file`, `write_file`, `edit_file`) are confined to the repository root via Go 1.22+'s `os.OpenRoot`. Paths are also cleaned with `filepath.Clean` as defense-in-depth.
- **Read-only / mutating separation** - read-only tools run concurrently (up to 4 workers); mutating tools run sequentially after all reads complete.
- **Bash with human-in-the-loop** - `run_bash` sets the working directory to the repo root but requires explicit `y` confirmation from the user before executing any command.
- **Per-turn timeout & cancellation** - each turn has a configurable timeout (default 5 min) and responds to `Ctrl+C` mid-turn.

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

### REPL Commands

| Input | Action |
|-------|--------|
| Any text | Send as a prompt to the agent |
| `/exit` or `/quit` | Terminate the session |
| `Ctrl+C` | Cancel the current turn mid-execution |

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-url` | `http://localhost:8000/v1` | OpenAI-compatible vLLM base URL |
| `-root` | `.` | Repository root directory for filesystem sandboxing |
| `-timeout` | `5m` | Per-turn timeout duration |
| `-max-context | `0` | Override context window size in tokens (`0` = auto-detect from vLLM `max_model_len`, fallback `32768`) |

## Requirements

- Go 1.22+ (for `os.OpenRoot` filesystem sandboxing)
- A running OpenAI-compatible inference server (e.g., vLLM on NVIDIA DGX Spark)
