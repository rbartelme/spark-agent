package agent

import (
	"context"
	"fmt"

	"github.com/rbartelme/spark-agent/internal/llm"
)

const (
	defaultMaxContext   = 32768
	pruneThresholdRatio = 0.80
)

// ModelClient abstracts the remote vLLM inference backend.
type ModelClient interface {
	ActiveModelInfo(ctx context.Context) (llm.ModelInfo, error)
	ChatStream(ctx context.Context, req llm.ChatRequest, onDelta func(string)) (llm.Message, llm.Usage, error)
}

// ToolExecutor abstracts tool schema discovery and batch execution.
type ToolExecutor interface {
	Schemas() []llm.ToolDefinition
	ExecuteBatch(ctx context.Context, calls []llm.ToolCall) []llm.Message
}

// Agent manages conversation state and executes tool-calling turns.
type Agent struct {
	client             ModelClient
	tools              ToolExecutor
	messages           []llm.Message
	maxSteps           int
	maxContextOverride int
	lastUsage          llm.Usage
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

// SetMaxContext overrides the auto-detected vLLM max_model_len if > 0.
func (a *Agent) SetMaxContext(tokens int) {
	a.maxContextOverride = tokens
}

// RunTurn processes a single user prompt through streaming completions and tool executions.
// If onDelta is non-nil, text tokens are streamed to it as they arrive.
func (a *Agent) RunTurn(ctx context.Context, userPrompt string, onDelta func(string)) (string, error) {
	info, err := a.client.ActiveModelInfo(ctx)
	if err != nil {
		return "", fmt.Errorf("resolve active model: %w", err)
	}

	budget := info.MaxModelLen
	if a.maxContextOverride > 0 {
		budget = a.maxContextOverride
	}
	if budget <= 0 {
		budget = defaultMaxContext
	}

	a.maybePruneHistory(budget)
	a.messages = append(a.messages, llm.Message{
		Role:    "user",
		Content: userPrompt,
	})

	for step := 0; step < a.maxSteps; step++ {
		if err := ctx.Err(); err != nil {
			return "", err
		}

		a.maybePruneHistory(budget)

		streamed := false
		reply, usage, err := a.client.ChatStream(ctx, llm.ChatRequest{
			Model:       info.ID,
			Messages:    a.messages,
			Tools:       a.tools.Schemas(),
			Temperature: 0.1,
		}, func(delta string) {
			streamed = true
			if onDelta != nil {
				onDelta(delta)
			}
		})
		if err != nil {
			return "", err
		}

		if usage.TotalTokens > 0 {
			a.lastUsage = usage
		}
		a.messages = append(a.messages, reply)

		if len(reply.ToolCalls) == 0 {
			if streamed && onDelta != nil {
				onDelta("\n")
			}
			return reply.Content, nil
		}

		if streamed && onDelta != nil {
			onDelta("\n")
		}

		for _, tc := range reply.ToolCalls {
			fmt.Printf("-> %s(%s)\n", tc.Function.Name, tc.Function.Arguments)
		}

		toolMessages := a.tools.ExecuteBatch(ctx, reply.ToolCalls)
		a.messages = append(a.messages, toolMessages...)
	}

	return "", fmt.Errorf("agent exceeded maximum tool iterations (%d)", a.maxSteps)
}

// maybePruneHistory drops the oldest complete user turn when usage reaches 80% of budget.
func (a *Agent) maybePruneHistory(budget int) {
	limit := int(float64(budget) * pruneThresholdRatio)
	if a.lastUsage.TotalTokens < limit || len(a.messages) <= 3 {
		return
	}

	// Keep messages[0] (system prompt) and cut at the second "user" message
	// so assistant tool_calls and tool responses stay paired.
	cutIdx := -1
	userCount := 0
	for i := 1; i < len(a.messages); i++ {
		if a.messages[i].Role == "user" {
			userCount++
			if userCount == 2 {
				cutIdx = i
				break
			}
		}
	}

	if cutIdx > 1 {
		dropped := cutIdx - 1
		a.messages = append(a.messages[:1], a.messages[cutIdx:]...)
		a.lastUsage.TotalTokens = 0
		fmt.Printf("[context] pruned %d old messages to stay within %d-token window\n", dropped, budget)
	}
}
