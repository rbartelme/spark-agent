package tools

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"github.com/rbartelme/spark-agent/internal/llm"
)

type toolHandler func(ctx context.Context, raw json.RawMessage) (string, error)

type registeredTool struct {
	def      llm.ToolDefinition
	readOnly bool
	run      toolHandler
}

// Registry manages sandboxed repository tools and dispatches model tool calls.
type Registry struct {
	root     *os.Root
	rootDir  string
	stdin    *bufio.Reader
	handlers map[string]registeredTool
	order    []string
}

// Register binds a strongly typed Go function to a tool name and JSON Schema.
func Register[T any](
	r *Registry,
	name string,
	description string,
	readOnly bool,
	rawSchema string,
	fn func(ctx context.Context, args T) (string, error),
) {
	var params map[string]any
	if err := json.Unmarshal([]byte(rawSchema), &params); err != nil {
		panic(fmt.Sprintf("invalid JSON schema for tool %s: %v", name, err))
	}

	r.handlers[name] = registeredTool{
		def: llm.ToolDefinition{
			Type: "function",
			Function: llm.FunctionSchema{
				Name:        name,
				Description: description,
				Parameters:  params,
			},
		},
		readOnly: readOnly,
		run: func(ctx context.Context, raw json.RawMessage) (string, error) {
			var args T
			if len(raw) > 0 {
				if err := json.Unmarshal(raw, &args); err != nil {
					return "", fmt.Errorf("invalid arguments for %s: %w", name, err)
				}
			}
			return fn(ctx, args)
		},
	}
	r.order = append(r.order, name)
}

// NewRegistry opens an OS-level filesystem sandbox rooted at rootDir and registers default tools.
func NewRegistry(rootDir string, in io.Reader) (*Registry, error) {
	root, err := os.OpenRoot(rootDir)
	if err != nil {
		return nil, fmt.Errorf("open root %s: %w", rootDir, err)
	}

	r := &Registry{
		root:     root,
		rootDir:  rootDir,
		stdin:    bufio.NewReader(in),
		handlers: make(map[string]registeredTool),
	}
	r.registerDefaults()
	return r, nil
}

// Close releases the underlying os.Root file descriptor.
func (r *Registry) Close() error {
	return r.root.Close()
}

// Schemas returns all registered tool definitions in deterministic registration order.
func (r *Registry) Schemas() []llm.ToolDefinition {
	defs := make([]llm.ToolDefinition, 0, len(r.order))
	for _, name := range r.order {
		defs = append(defs, r.handlers[name].def)
	}
	return defs
}

// ExecuteBatch runs read-only tools concurrently (up to 4 workers) and mutating tools sequentially,
// preserving exact tool_call_id ordering in the returned messages.
func (r *Registry) ExecuteBatch(ctx context.Context, calls []llm.ToolCall) []llm.Message {
	results := make([]llm.Message, len(calls))
	sem := make(chan struct{}, 4)
	var wg sync.WaitGroup

	for i, call := range calls {
		tool, exists := r.handlers[call.Function.Name]
		if !exists {
			results[i] = llm.Message{
				Role:       "tool",
				ToolCallID: call.ID,
				Content:    fmt.Sprintf("Error: unknown tool %q", call.Function.Name),
			}
			continue
		}

		if tool.readOnly {
			wg.Add(1)
			go func(idx int, tc llm.ToolCall, h toolHandler) {
				defer wg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()

				results[idx] = r.invoke(ctx, tc, h)
			}(i, call, tool.run)
		} else {
			// Wait for any in-flight reads before mutating files or running commands.
			wg.Wait()
			results[i] = r.invoke(ctx, call, tool.run)
		}
	}

	wg.Wait()
	return results
}

func (r *Registry) invoke(ctx context.Context, call llm.ToolCall, h toolHandler) llm.Message {
	out, err := h(ctx, json.RawMessage(call.Function.Arguments))
	if err != nil {
		out = fmt.Sprintf("Error: %v", err)
	}
	return llm.Message{
		Role:       "tool",
		ToolCallID: call.ID,
		Content:    out,
	}
}

func (r *Registry) registerDefaults() {
	Register(r, "list_files", "List relative file paths in the repository (skips .git).", true,
		`{"type":"object","properties":{"subdir":{"type":"string","description":"Optional subdirectory, defaults to '.'"}}}`,
		func(_ context.Context, args struct {
			Subdir string `json:"subdir"`
		}) (string, error) {
			start := "."
			if strings.TrimSpace(args.Subdir) != "" {
				start = filepath.Clean(args.Subdir)
			}
			var files []string
			err := fs.WalkDir(r.root.FS(), start, func(path string, d fs.DirEntry, err error) error {
				if err != nil {
					return err
				}
				if d.IsDir() && (d.Name() == ".git" || d.Name() == "node_modules" || d.Name() == "target") {
					return filepath.SkipDir
				}
				if !d.IsDir() {
					files = append(files, path)
				}
				return nil
			})
			if err != nil {
				return "", err
			}
			if len(files) == 0 {
				return "(no files found)", nil
			}
			return strings.Join(files, "\n"), nil
		},
	)

	Register(r, "read_file", "Read the full contents of a file inside the repository.", true,
		`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`,
		func(_ context.Context, args struct {
			Path string `json:"path"`
		}) (string, error) {
			data, err := r.root.ReadFile(filepath.Clean(args.Path))
			if err != nil {
				return "", err
			}
			return string(data), nil
		},
	)

	Register(r, "write_file", "Create or overwrite a file inside the repository.", false,
		`{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}`,
		func(_ context.Context, args struct {
			Path    string `json:"path"`
			Content string `json:"content"`
		}) (string, error) {
			clean := filepath.Clean(args.Path)
			if dir := filepath.Dir(clean); dir != "." {
				if err := r.root.MkdirAll(dir, 0o755); err != nil {
					return "", err
				}
			}
			if err := r.root.WriteFile(clean, []byte(args.Content), 0o644); err != nil {
				return "", err
			}
			return fmt.Sprintf("Wrote %d bytes to %s", len(args.Content), clean), nil
		},
	)

	Register(r, "edit_file", "Replace an exact, unique text snippet in an existing file.", false,
		`{"type":"object","properties":{"path":{"type":"string"},"old_text":{"type":"string"},"new_text":{"type":"string"}},"required":["path","old_text","new_text"]}`,
		func(_ context.Context, args struct {
			Path    string `json:"path"`
			OldText string `json:"old_text"`
			NewText string `json:"new_text"`
		}) (string, error) {
			clean := filepath.Clean(args.Path)
			data, err := r.root.ReadFile(clean)
			if err != nil {
				return "", err
			}
			content := string(data)
			count := strings.Count(content, args.OldText)
			if count == 0 {
				return "", fmt.Errorf("old_text not found in %s", clean)
			}
			if count > 1 {
				return "", fmt.Errorf("old_text matched %d times in %s; provide more surrounding context", count, clean)
			}
			updated := strings.Replace(content, args.OldText, args.NewText, 1)
			if err := r.root.WriteFile(clean, []byte(updated), 0o644); err != nil {
				return "", err
			}
			return fmt.Sprintf("Updated %s", clean), nil
		},
	)

	Register(r, "run_bash", "Run a shell command in the repository root after user confirmation.", false,
		`{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}`,
		func(ctx context.Context, args struct {
			Command string `json:"command"`
		}) (string, error) {
			fmt.Printf("\n[Allow bash: `%s`? y/N]: ", args.Command)
			line, err := r.stdin.ReadString('\n')
			if err != nil && err != io.EOF {
				return "", err
			}
			if !strings.EqualFold(strings.TrimSpace(line), "y") {
				return "User denied command execution.", nil
			}
			cmd := exec.CommandContext(ctx, "bash", "-c", args.Command)
			cmd.Dir = r.rootDir
			out, err := cmd.CombinedOutput()
			if err != nil {
				return fmt.Sprintf("Exit error: %v\nOutput:\n%s", err, string(out)), nil
			}
			if len(out) == 0 {
				return "(command succeeded with no output)", nil
			}
			return string(out), nil
		},
	)
}
