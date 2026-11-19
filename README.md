# spark-agent

A minimal Go CLI harness for running a tool-calling coding agent against a local OpenAI-compatible vLLM endpoint (e.g., NVIDIA DGX Spark). The agent operates inside a sandboxed repository root, with filesystem operations confined via `os.OpenRoot` and shell commands gated behind interactive user confirmation.

## Architecture

```
spark-agent/
├── cmd/spark-agent/main.go   # CLI entry: flag parsing, REPL loop, signal handling
├── internal/
│   ├── agent/loop.go         # Turn state machine: prompt -> model -> tool calls -> repeat
│   ├── llm/client.go         # OpenAI-compatible HTTP client (/v1/models, /v1/chat/completions)
│   └── tools/registry.go     # Typed tool definitions, os.Root sandboxing, batch execution
```

### Components

| Package | Role |
|---------|------|
| `cmd/spark-agent` | Parses flags, starts the REPL, manages per-turn timeouts and Ctrl+C cancellation |
| `internal/agent` | `Agent` struct holds conversation history, dispatches chat requests, and loops on tool calls (max 15 steps per turn) |
| `internal/llm` | HTTP client for OpenAI-compatible endpoints; discovers the active model and sends chat completions with tool schemas |
| `internal/tools` | `Registry` manages tool definitions, enforces filesystem sandboxing, and executes tool calls in batch |

## Features

- **Tool-calling agent loop** - sends user prompts to the model, executes returned tool calls, feeds results back, and repeats until the model responds without tool calls.
- **Filesystem sandboxing** - all file operations (`list_files`, `read_file`, `write_file`, `edit_file`) are confined to the repository root via Go 1.22+'s `os.OpenRoot`. Paths are also cleaned with `filepath.Clean` as defense-in-depth.
- **Read-only / mutating separation** - read-only tools run concurrently (up to 4 workers); mutating tools run sequentially after all reads complete.
- **Bash with human-in-the-loop** - `run_bash` sets the working directory to the repo root but requires explicit `y` confirmation from the user before executing any command.
- **Per-turn timeout & cancellation** - each turn has a configurable timeout (default 5 min) and responds to `Ctrl+C` mid-turn.

## Tools

| Tool | Read-Only | Description |
|------|-----------|-------------|
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

# Custom per-turn timeout
./spark-agent -timeout 10m
```

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

## Requirements

- Go 1.22+ (for `os.OpenRoot` filesystem sandboxing)
- A running OpenAI-compatible inference server (e.g., vLLM on NVIDIA DGX Spark)
