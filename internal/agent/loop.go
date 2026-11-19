package agent

import (
	"context"
	"fmt"

	"github.com/rbartelme/spark-agent/internal/llm"
)

// ModelClient abstracts the remote vLLM inference backend.
type ModelClient interface {
	ActiveModel(ctx context.Context) (string, error)
	Chat(ctx context.Context, req llm.ChatRequest) (llm.Message, error)
}

// ToolExecutor abstracts tool schema discovery and batch execution.
type ToolExecutor interface {
	Schemas() []llm.ToolDefinition
	ExecuteBatch(ctx context.Context, calls []llm.ToolCall) []llm.Message
}

// Agent manages conversation state and executes tool-calling turns.
type Agent struct {
	client   ModelClient
	tools    ToolExecutor
	messages []llm.Message
	maxSteps int
}

// New creates an Agent initialized with a system prompt.
func New(client ModelClient, tools ToolExecutor, systemPrompt string) *Agent {
	return &Agent{
		client:   client,
		tools:    tools,
		maxSteps: 15,
		messages: []llm.Message{
			{Role: "system", Content: systemPrompt},
		},
	}
}

// RunTurn processes a single user prompt through model completions and tool executions.
func (a *Agent) RunTurn(ctx context.Context, userPrompt string) (string, error) {
	model, err := a.client.ActiveModel(ctx)
	if err != nil {
		return "", fmt.Errorf("resolve active model: %w", err)
	}

	a.messages = append(a.messages, llm.Message{
		Role:    "user",
		Content: userPrompt,
	})

	for step := 0; step < a.maxSteps; step++ {
		if err := ctx.Err(); err != nil {
			return "", err
		}

		reply, err := a.client.Chat(ctx, llm.ChatRequest{
			Model:       model,
			Messages:    a.messages,
			Tools:       a.tools.Schemas(),
			Temperature: 0.1,
		})
		if err != nil {
			return "", err
		}

		a.messages = append(a.messages, reply)

		if len(reply.ToolCalls) == 0 {
			return reply.Content, nil
		}

		for _, tc := range reply.ToolCalls {
			fmt.Printf("-> %s(%s)\n", tc.Function.Name, tc.Function.Arguments)
		}

		toolMessages := a.tools.ExecuteBatch(ctx, reply.ToolCalls)
		a.messages = append(a.messages, toolMessages...)
	}

	return "", fmt.Errorf("agent exceeded maximum tool iterations (%d)", a.maxSteps)
}
