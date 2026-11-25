package vllm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// ModelConfig holds the configuration for a single model in the zoo.
type ModelConfig struct {
	Service       string
	Port          int
	ModelID       string
	IsSidecar     bool
	HighMemory    bool
	ComposeFile   string
}

// ModelStatus represents the runtime state of a model.
type ModelStatus struct {
	Service string
	Port    int
	State   string // "STOPPED", "SLEEPING (L2)", "AWAKE"
}

// Manager coordinates the vLLM model zoo lifecycle.
type Manager struct {
	mu          sync.Mutex
	registry    map[string]ModelConfig
	composeFile string
	httpClient  *http.Client
}

// NewManager creates a Manager with the default model registry.
func NewManager(composeFile string) *Manager {
	m := &Manager{
		registry: make(map[string]ModelConfig),
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	}

	// Resolve compose file path
	m.composeFile = resolveComposePath(composeFile)

	// Register models
	m.registry["sidecar"] = ModelConfig{
		Service:   "sidecar-7b",
		Port:      8001,
		ModelID:   "Qwen/Qwen2.5-Coder-7B-Instruct",
		IsSidecar: true,
	}
	m.registry["Qwen/Qwen2.5-Coder-7B-Instruct"] = m.registry["sidecar"]

	m.registry["gemma4"] = ModelConfig{
		Service: "gemma4",
		Port:    8002,
		ModelID: "google/gemma-4-26B-A4B",
	}
	m.registry["google/gemma-4-26B-A4B"] = m.registry["gemma4"]

	m.registry["qwen3-coder"] = ModelConfig{
		Service:    "qwen3-coder",
		Port:       8003,
		ModelID:    "Qwen/Qwen3-Coder-Next-FP8",
		HighMemory: true,
	}
	m.registry["Qwen/Qwen3-Coder-Next-FP8"] = m.registry["qwen3-coder"]

	m.registry["qwen36-35b"] = ModelConfig{
		Service: "qwen36-35b",
		Port:    8004,
		ModelID: "Qwen/Qwen3.6-35B-A3B",
	}
	m.registry["Qwen/Qwen3.6-35B-A3B"] = m.registry["qwen36-35b"]

	return m
}

// resolveComposePath determines the docker-compose.yml path.
func resolveComposePath(override string) string {
	if override != "" {
		if _, err := os.Stat(override); err == nil {
			return override
		}
		return override
	}

	// Check env var
	if env := os.Getenv("SPARK_VLLM_COMPOSE"); env != "" {
		if _, err := os.Stat(env); err == nil {
			return env
		}
	}

	// Walk up from working directory
	cwd, err := os.Getwd()
	if err == nil {
		dir := cwd
		for {
			candidate := filepath.Join(dir, "deploy", "vllm", "docker-compose.yml")
			if _, err := os.Stat(candidate); err == nil {
				return candidate
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}

	// Walk up from executable
	exe, err := os.Executable()
	if err == nil {
		dir := filepath.Dir(exe)
		for {
			candidate := filepath.Join(dir, "deploy", "vllm", "docker-compose.yml")
			if _, err := os.Stat(candidate); err == nil {
				return candidate
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}

	// Default relative path
	return "deploy/vllm/docker-compose.yml"
}

// resolveModel looks up a model by name or alias.
func (m *Manager) resolveModel(modelOrAlias string) (ModelConfig, error) {
	cfg, ok := m.registry[modelOrAlias]
	if !ok {
		return ModelConfig{}, fmt.Errorf("unknown model or alias: %s", modelOrAlias)
	}
	return cfg, nil
}

// isContainerRunning checks if a docker container is running.
func isContainerRunning(service string) bool {
	cmd := exec.Command("docker", "inspect", "-f", "{{.State.Running}}", "vllm-"+service)
	out, err := cmd.Output()
	if err != nil {
		return false
	}
	return strings.TrimSpace(string(out)) == "true"
}

// isSleeping checks if a model is currently sleeping.
func (m *Manager) isSleeping(port int) (bool, error) {
	resp, err := m.httpClient.Get(fmt.Sprintf("http://localhost:%d/is_sleeping", port))
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return false, fmt.Errorf("is_sleeping returned %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 256))
	if err != nil {
		return false, err
	}

	var result struct {
		IsSleeping bool `json:"is_sleeping"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return false, fmt.Errorf("failed to parse is_sleeping response: %w", err)
	}
	return result.IsSleeping, nil
}

// healthCheck checks if a model endpoint is healthy.
func (m *Manager) healthCheck(port int) bool {
	resp, err := m.httpClient.Get(fmt.Sprintf("http://localhost:%d/health", port))
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// sleepModel sends a Level 2 sleep request to a model.
func (m *Manager) sleepModel(port int) error {
	resp, err := m.httpClient.Post(
		fmt.Sprintf("http://localhost:%d/sleep?level=2&mode=wait", port),
		"application/json",
		nil,
	)
	if err != nil {
		return fmt.Errorf("sleep request failed: %w", err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	return nil
}

// wakeModel performs the Level 2 wake sequence.
func (m *Manager) wakeModel(port int) error {
	// Step 1: POST /wake_up?tags=weights
	resp, err := m.httpClient.Post(
		fmt.Sprintf("http://localhost:%d/wake_up?tags=weights", port),
		"application/json",
		nil,
	)
	if err != nil {
		return fmt.Errorf("wake_up weights failed: %w", err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)

	// Step 2: POST /collective_rpc with {"method":"reload_weights"}
	rpcBody := []byte(`{"method":"reload_weights"}`)
	resp, err = m.httpClient.Post(
		fmt.Sprintf("http://localhost:%d/collective_rpc", port),
		"application/json",
		bytes.NewReader(rpcBody),
	)
	if err != nil {
		return fmt.Errorf("collective_rpc failed: %w", err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)

	// Step 3: POST /wake_up?tags=kv_cache
	resp, err = m.httpClient.Post(
		fmt.Sprintf("http://localhost:%d/wake_up?tags=kv_cache", port),
		"application/json",
		nil,
	)
	if err != nil {
		return fmt.Errorf("wake_up kv_cache failed: %w", err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)

	return nil
}

// startContainer starts a single model container and waits for it to be ready.
func (m *Manager) startContainer(ctx context.Context, cfg ModelConfig) error {
	cmd := exec.CommandContext(ctx, "docker", "compose", "-f", m.composeFile, "up", "-d", cfg.Service)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start container: %w", err)
	}

	// Poll /health until ready (max 120 seconds)
	maxRetries := 120
	for i := 0; i < maxRetries; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if m.healthCheck(cfg.Port) {
			return nil
		}
		time.Sleep(1 * time.Second)
	}

	return fmt.Errorf("container %s did not become healthy within %ds", cfg.Service, maxRetries)
}

// EnsureReady ensures the requested model is awake and returns its base URL and model ID.
// It handles unified memory enforcement, cold-start fallback, and Level 2 wake sequences.
func (m *Manager) EnsureReady(ctx context.Context, modelOrAlias string) (baseURL string, modelID string, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	cfg, err := m.resolveModel(modelOrAlias)
	if err != nil {
		return "", "", err
	}

	cfg.ComposeFile = m.composeFile

	// Check if container is running
	if !isContainerRunning(cfg.Service) {
		// Cold-start fallback: start the container
		fmt.Printf("Container %s not running, starting...\n", cfg.Service)
		if err := m.startContainer(ctx, cfg); err != nil {
			return "", "", fmt.Errorf("cold start failed for %s: %w", cfg.Service, err)
		}
	}

	// Check if already awake
	sleeping, err := m.isSleeping(cfg.Port)
	if err != nil {
		// If we can't check sleeping status, the container might not be ready yet
		if !m.healthCheck(cfg.Port) {
			fmt.Printf("Container %s not healthy, starting...\n", cfg.Service)
			if err := m.startContainer(ctx, cfg); err != nil {
				return "", "", fmt.Errorf("cold start failed for %s: %w", cfg.Service, err)
			}
		}
	}

	if !sleeping {
		// Already awake, sleep ALL other primary models (including HighMemory)
		for alias, other := range m.registry {
			if alias == modelOrAlias {
				continue
			}
			if !other.IsSidecar {
				if isContainerRunning(other.Service) {
					if s, err := m.isSleeping(other.Port); err == nil && !s {
						fmt.Printf("Sleeping %s (%s) to free memory for %s\n", other.Service, other.ModelID, cfg.ModelID)
						_ = m.sleepModel(other.Port)
					}
				}
			}
		}
		return fmt.Sprintf("http://localhost:%d/v1", cfg.Port), cfg.ModelID, nil
	}

	// Level 2 wake sequence
	fmt.Printf("Waking %s (%s) from Level 2 sleep...\n", cfg.Service, cfg.ModelID)
	if err := m.wakeModel(cfg.Port); err != nil {
		return "", "", fmt.Errorf("wake sequence failed for %s: %w", cfg.Service, err)
	}

	// Sleep ALL other primary models (including HighMemory) after wake
	for alias, other := range m.registry {
		if alias == modelOrAlias {
			continue
		}
		if !other.IsSidecar {
			if isContainerRunning(other.Service) {
				if s, err := m.isSleeping(other.Port); err == nil && !s {
					fmt.Printf("Sleeping %s (%s) to free memory for %s\n", other.Service, other.ModelID, cfg.ModelID)
					_ = m.sleepModel(other.Port)
				}
			}
		}
	}

	return fmt.Sprintf("http://localhost:%d/v1", cfg.Port), cfg.ModelID, nil
}

// SleepAll sleeps all awake models in the zoo.
func (m *Manager) SleepAll() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, cfg := range m.registry {
		if isContainerRunning(cfg.Service) {
			if s, err := m.isSleeping(cfg.Port); err == nil && !s {
				fmt.Printf("Sleeping %s...\n", cfg.Service)
				if err := m.sleepModel(cfg.Port); err != nil {
					fmt.Printf("Warning: failed to sleep %s: %v\n", cfg.Service, err)
				}
			}
		}
	}
	return nil
}

// DownAll stops all model containers.
func (m *Manager) DownAll() error {
	cmd := exec.Command("docker", "compose", "-f", m.composeFile, "down")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// Status returns the status of all models in the zoo.
func (m *Manager) Status() []ModelStatus {
	m.mu.Lock()
	defer m.mu.Unlock()

	seen := make(map[string]bool)
	var statuses []ModelStatus
	for _, cfg := range m.registry {
		if seen[cfg.Service] {
			continue
		}
		seen[cfg.Service] = true

		status := ModelStatus{
			Service: cfg.Service,
			Port:    cfg.Port,
		}

		if !m.healthCheck(cfg.Port) {
			status.State = "STOPPED"
		} else {
			s, err := m.isSleeping(cfg.Port)
			if err != nil {
				status.State = "UNKNOWN"
			} else if s {
				status.State = "SLEEPING (L2)"
			} else {
				status.State = "AWAKE"
			}
		}

		statuses = append(statuses, status)
	}

	return statuses
}

// AvailableModels returns a list of all registered model aliases.
func (m *Manager) AvailableModels() []string {
	m.mu.Lock()
	defer m.mu.Unlock()

	var models []string
	for alias := range m.registry {
		models = append(models, alias)
	}
	return models
}

// GetModelConfig returns the config for a model by name or alias.
func (m *Manager) GetModelConfig(modelOrAlias string) (ModelConfig, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	cfg, ok := m.registry[modelOrAlias]
	if !ok {
		return ModelConfig{}, fmt.Errorf("unknown model or alias: %s", modelOrAlias)
	}
	return cfg, nil
}
