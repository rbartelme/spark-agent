# spark-agent
minimal golang CLI harness for directly connecting to agents hosted on spark dgx

```tree
spark-agent/
├── cmd/spark-agent/main.go   # Flag parsing, signal context, REPL loop
└── internal/
    ├── agent/loop.go         # Turn state machine, message history, cancellation
    ├── llm/client.go         # OpenAI-compatible HTTP client & /v1/models check
    └── tools/registry.go     # Typed tool definitions, os.Root sandboxing, handlers
```
