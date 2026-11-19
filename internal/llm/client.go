package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Message represents a single turn in an OpenAI-style chat history.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string `json:"tool_call_id,omitempty"`
}

// ToolCall represents a call to a tool in the chat history.
type ToolCall struct {
	ID string `json:"id"`
	Type string `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall holds the target tool name and raw JSON argument payload.
type FunctionCall struct {
	Name string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolDeinition describes and available tool passed to vLLM
type ToolDefinition struct {
	Type string `json:"type"`
	Function FunctionSchema `json:"function"`
}

// FunctionSchema defines the name, description, and JSON Schema parameters of a tool.
type FunctionSchema struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

//ChatRequest payload sent to /v1/chat/completions endpoint.
type ChatRequest struct {
	Model            string           `json:"model"`
	Messages         []Message        `json:"messages"`
	Tools            []ToolDefinition `json:"tools,omitempty"`
	Temperature      float64          `json:"temperature"`
}

// Client communicates with OpenAI compatible vLLM enpoint over HTTP.
type Client struct {
	baseURL    string
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

// ActiveModel queries vLLM /v1/models to discover the currently active model on DGX spark. Returns empty string if none are active.
func (c *Client) ActiveModel(ctx context.Context) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/models", nil)
	if err != nil {
		return "", fmt.Errorf("Model request failed: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("Fetch active model failed: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return "", fmt.Errorf("models endpoint returned %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return "", fmt.Errorf("Failed to decode models response: %w", err)
	}
	if len(payload.Data) == 0 {
		return "", fmt.Errorf("no models served at %s", c.baseURL) // No active model
	}
	return payload.Data[0].ID, nil
}

// Chat sends a completion request to the vLLM /v1/chat/completions endpoint and returns the response body.
func (c *Client) Chat(ctx context.Context, chatReq ChatRequest) (Message, error) {
	encoded, err := json.Marshal(chatReq)
	if err != nil {
		return Message{}, fmt.Errorf("Failed to encode chat request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(encoded))
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
		Choises []struct {
			Message Message `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return Message{}, fmt.Errorf("Failed to decode chat response: %w", err)
	}
	if len(chatResp.Choises) == 0 {
		return Message{}, fmt.Errorf("no choices returned from chat endpoint")
	}
	return chatResp.Choises[0].Message, nil
}