package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Message represents a single turn in an OpenAI-style chat history.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ToolCall represents a call to a tool in the chat history.
type ToolCall struct {
	Index    int          `json:"index,omitempty"`
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall holds the target tool name and raw JSON argument payload.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolDefinition describes an available tool passed to vLLM.
type ToolDefinition struct {
	Type     string         `json:"type"`
	Function FunctionSchema `json:"function"`
}

// FunctionSchema defines the name, description, and JSON Schema parameters of a tool.
type FunctionSchema struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// StreamOptions configures streaming behavior on OpenAI-compatible endpoints.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// ChatRequest payload sent to /v1/chat/completions endpoint.
type ChatRequest struct {
	Model         string           `json:"model"`
	Messages      []Message        `json:"messages"`
	Tools         []ToolDefinition `json:"tools,omitempty"`
	Temperature   float64          `json:"temperature"`
	Stream        bool             `json:"stream,omitempty"`
	StreamOptions *StreamOptions   `json:"stream_options,omitempty"`
}

// ModelInfo holds the active model ID and context window size from vLLM /v1/models.
type ModelInfo struct {
	ID          string
	MaxModelLen int
}

// Usage tracks token counts returned by vLLM at the end of a completion.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Client communicates with OpenAI compatible vLLM endpoint over HTTP.
type Client struct {
	mu         sync.RWMutex
	baseURL    string
	modelID    string
	maxModelLen int
	httpClient *http.Client
}

// NewClient creates a new Client with the given base URL and timeout.
func NewClient(baseURL string, timeout time.Duration) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

// SetBaseURL updates the client's active base URL.
func (c *Client) SetBaseURL(baseURL string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.baseURL = strings.TrimRight(baseURL, "/")
	c.modelID = ""
	c.maxModelLen = 0
}

// SetModelInfo updates the cached model ID and max model length.
func (c *Client) SetModelInfo(modelID string, maxModelLen int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.modelID = modelID
	c.maxModelLen = maxModelLen
}

// BaseURL returns the current base URL.
func (c *Client) BaseURL() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.baseURL
}

// ModelID returns the cached model ID.
func (c *Client) ModelID() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.modelID
}

// MaxModelLen returns the cached max model length.
func (c *Client) MaxModelLen() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.maxModelLen
}

// ActiveModel queries vLLM /v1/models to discover the currently active model ID on DGX Spark.
func (c *Client) ActiveModel(ctx context.Context) (string, error) {
	info, err := c.ActiveModelInfo(ctx)
	if err != nil {
		return "", err
	}
	return info.ID, nil
}

// ActiveModelInfo queries vLLM /v1/models for both the active model ID and max_model_len.
func (c *Client) ActiveModelInfo(ctx context.Context) (ModelInfo, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL()+"/models", nil)
	if err != nil {
		return ModelInfo{}, fmt.Errorf("Model request failed: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return ModelInfo{}, fmt.Errorf("Fetch active model failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return ModelInfo{}, fmt.Errorf("models endpoint returned %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var payload struct {
		Data []struct {
			ID          string `json:"id"`
			MaxModelLen int    `json:"max_model_len"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return ModelInfo{}, fmt.Errorf("Failed to decode models response: %w", err)
	}
	if len(payload.Data) == 0 {
		return ModelInfo{}, fmt.Errorf("no models served at %s", c.baseURL)
	}
	return ModelInfo{
		ID:          payload.Data[0].ID,
		MaxModelLen: payload.Data[0].MaxModelLen,
	}, nil
}

// Chat sends a non-streaming completion request to the vLLM /v1/chat/completions endpoint.
func (c *Client) Chat(ctx context.Context, chatReq ChatRequest) (Message, error) {
	chatReq.Stream = false
	chatReq.StreamOptions = nil

	encoded, err := json.Marshal(chatReq)
	if err != nil {
		return Message{}, fmt.Errorf("Failed to encode chat request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL()+"/chat/completions", bytes.NewReader(encoded))
	if err != nil {
		return Message{}, fmt.Errorf("Failed to create chat request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return Message{}, fmt.Errorf("Chat request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return Message{}, fmt.Errorf("chat endpoint returned %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var chatResp struct {
		Choices []struct {
			Message Message `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return Message{}, fmt.Errorf("Failed to decode chat response: %w", err)
	}
	if len(chatResp.Choices) == 0 {
		return Message{}, fmt.Errorf("no choices returned from chat endpoint")
	}
	return chatResp.Choices[0].Message, nil
}

// ChatStream sends a streaming completion request to /v1/chat/completions, invoking onDelta
// as text tokens arrive and returning the reassembled assistant Message and token Usage.
func (c *Client) ChatStream(
	ctx context.Context,
	chatReq ChatRequest,
	onDelta func(string),
) (Message, Usage, error) {
	chatReq.Stream = true
	chatReq.StreamOptions = &StreamOptions{IncludeUsage: true}

	encoded, err := json.Marshal(chatReq)
	if err != nil {
		return Message{}, Usage{}, fmt.Errorf("Failed to encode chat request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL()+"/chat/completions", bytes.NewReader(encoded))
	if err != nil {
		return Message{}, Usage{}, fmt.Errorf("Failed to create chat request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return Message{}, Usage{}, fmt.Errorf("Chat request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return Message{}, Usage{}, fmt.Errorf("chat endpoint returned %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	var contentBuf strings.Builder
	toolMap := make(map[int]*ToolCall)
	maxToolIdx := -1
	var finalUsage Usage

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}

		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content   string     `json:"content,omitempty"`
					ToolCalls []ToolCall `json:"tool_calls,omitempty"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *Usage `json:"usage,omitempty"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return Message{}, Usage{}, fmt.Errorf("Failed to decode stream chunk: %w", err)
		}

		if chunk.Usage != nil {
			finalUsage = *chunk.Usage
		}
		if len(chunk.Choices) == 0 {
			continue
		}

		delta := chunk.Choices[0].Delta
		if delta.Content != "" {
			contentBuf.WriteString(delta.Content)
			if onDelta != nil {
				onDelta(delta.Content)
			}
		}

		for _, tc := range delta.ToolCalls {
			existing, ok := toolMap[tc.Index]
			if !ok {
				existing = &ToolCall{Index: tc.Index, Type: "function"}
				toolMap[tc.Index] = existing
				if tc.Index > maxToolIdx {
					maxToolIdx = tc.Index
				}
			}
			if tc.ID != "" {
				existing.ID = tc.ID
			}
			if tc.Type != "" {
				existing.Type = tc.Type
			}
			if tc.Function.Name != "" {
				existing.Function.Name += tc.Function.Name
			}
			if tc.Function.Arguments != "" {
				existing.Function.Arguments += tc.Function.Arguments
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return Message{}, Usage{}, fmt.Errorf("Stream read failed: %w", err)
	}

	var assembledTools []ToolCall
	for i := 0; i <= maxToolIdx; i++ {
		if tc, ok := toolMap[i]; ok {
			assembledTools = append(assembledTools, *tc)
		}
	}

	return Message{
		Role:      "assistant",
		Content:   contentBuf.String(),
		ToolCalls: assembledTools,
	}, finalUsage, nil
}
