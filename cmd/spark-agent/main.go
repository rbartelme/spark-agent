package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/rbartelme/spark-agent/internal/agent"
	"github.com/rbartelme/spark-agent/internal/cli"
	"github.com/rbartelme/spark-agent/internal/llm"
	"github.com/rbartelme/spark-agent/internal/tools"
	"github.com/rbartelme/spark-agent/internal/vllm"
)

const defaultSystemPrompt = `You are a concise coding agent editing a local repository in place.
Inspect files with list_files and read_file before modifying code.
Prefer edit_file for surgical changes to existing files and write_file for new files.
Verify changes with run_bash when appropriate.
No unicode characters or emojis in your responses. Use only ASCII characters.`

func main() {
	baseURL := flag.String("url", "http://localhost:8000/v1", "OpenAI-compatible vLLM base URL")
	model := flag.String("model", "", "Model name or alias from the zoo (e.g. sidecar, qwen3-coder, gemma4)")
	composeFile := flag.String("compose", "", "Path to docker-compose.yml for the model zoo")
	rootDir := flag.String("root", ".", "Repository root directory to sandbox file operations")
	timeout := flag.Duration("timeout", 5*time.Minute, "Per-turn timeout")
	maxContext := flag.Int("max-context", 0, "Override context window size in tokens (0 = auto-detect from vLLM)")
	flag.Parse()

	// Handle "zoo" subcommands from positional args before anything else
	args := flag.Args()
	if len(args) > 0 && args[0] == "zoo" {
		mgr := vllm.NewManager(*composeFile)
		if len(args) >= 2 {
			switch args[1] {
			case "up":
				// Sleep all other models and ensure the zoo is ready
				mgr.SleepAll()
				fmt.Println("Model zoo started.")
			case "down":
				if err := mgr.DownAll(); err != nil {
					fmt.Fprintf(os.Stderr, "Failed to stop all models: %v\n", err)
					os.Exit(1)
				}
				fmt.Println("All models stopped.")
			case "status":
				statuses := mgr.Status()
				fmt.Printf("%-15s %-8s %-20s\n", "SERVICE", "PORT", "STATE")
				fmt.Printf("%-15s %-8s %-20s\n", "-------", "----", "-----")
				for _, s := range statuses {
					fmt.Printf("%-15s %-8d %-20s\n", s.Service, s.Port, s.State)
				}
			default:
				fmt.Fprintf(os.Stderr, "Unknown zoo command: %s\nUsage: spark-agent zoo [up|down|status]\n", args[1])
				os.Exit(1)
			}
			os.Exit(0)
		} else {
			fmt.Fprintf(os.Stderr, "Usage: spark-agent zoo [up|down|status]\n")
			os.Exit(1)
		}
	}

	registry, err := tools.NewRegistry(*rootDir, os.Stdin)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize tool registry: %v\n", err)
		os.Exit(1)
	}
	defer registry.Close()

	var client *llm.Client
	mgr := vllm.NewManager(*composeFile)

	if *model != "" {
		baseURLStr, modelID, err := mgr.EnsureReady(context.Background(), *model)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to ensure model %s is ready: %v\n", *model, err)
			os.Exit(1)
		}
		client = llm.NewClient(baseURLStr, *timeout)
		client.SetModelInfo(modelID, 0)
		fmt.Printf("Using model zoo model: %s -> %s\n", *model, baseURLStr)
	} else {
		client = llm.NewClient(*baseURL, *timeout)
	}

	bot := agent.New(client, registry, defaultSystemPrompt)
	if *maxContext > 0 {
		bot.SetMaxContext(*maxContext)
	}

	checkCtx, cancelCheck := context.WithTimeout(context.Background(), 5*time.Second)
	info, err := client.ActiveModelInfo(checkCtx)
	cancelCheck()
	if err != nil {
		fmt.Printf("Warning: could not reach %s (%v)\n", client.BaseURL(), err)
	} else {
		ctxLimit := info.MaxModelLen
		if *maxContext > 0 {
			ctxLimit = *maxContext
		}
		fmt.Printf("Connected to %s | Active model: %s | Context: %d tokens | Sandbox: %s\n", client.BaseURL(), info.ID, ctxLimit, *rootDir)
	}
	fmt.Println("Type a prompt, /exit to quit, /model <name> to switch, or press Ctrl+C during a turn to cancel.")

	promptReader := cli.NewPromptReader()

	for {
		fmt.Println()
		rawInput, err := promptReader.ReadPrompt()
		if err != nil {
			if !errors.Is(err, io.EOF) {
				fmt.Fprintf(os.Stderr, "Input error: %v\n", err)
			}
			break
		}

		prompt := strings.TrimSpace(rawInput)
		if prompt == "" {
			continue
		}

		// Handle /model subcommands
		if strings.HasPrefix(prompt, "/model") {
			args := strings.TrimSpace(strings.TrimPrefix(prompt, "/model"))
			args = strings.TrimSpace(args)
			switch args {
			case "up", "":
				// Default: ensure the current model is ready
				if *model != "" {
					baseURLStr, modelID, err := mgr.EnsureReady(context.Background(), *model)
					if err != nil {
						fmt.Fprintf(os.Stderr, "Failed to ensure model %s is ready: %v\n", *model, err)
						continue
					}
					client.SetBaseURL(baseURLStr)
					client.SetModelInfo(modelID, 0)
					fmt.Printf("Model %s is ready at %s\n", *model, baseURLStr)
				} else {
					fmt.Println("No model specified. Use /model <name> to switch.")
				}
			case "down":
				if err := mgr.DownAll(); err != nil {
					fmt.Fprintf(os.Stderr, "Failed to stop all models: %v\n", err)
				} else {
					fmt.Println("All models stopped.")
				}
			case "status":
				statuses := mgr.Status()
				fmt.Printf("%-15s %-8s %-20s\n", "SERVICE", "PORT", "STATE")
				fmt.Printf("%-15s %-8s %-20s\n", "-------", "----", "-----")
				for _, s := range statuses {
					fmt.Printf("%-15s %-8d %-20s\n", s.Service, s.Port, s.State)
				}
			case "list":
				models := mgr.AvailableModels()
				fmt.Println("Available models:")
				for _, m := range models {
					fmt.Printf("  %s\n", m)
				}
			default:
				// Treat as a model name to switch to
				baseURLStr, modelID, err := mgr.EnsureReady(context.Background(), args)
				if err != nil {
					fmt.Fprintf(os.Stderr, "Failed to ensure model %s is ready: %v\n", args, err)
					continue
				}
				client.SetBaseURL(baseURLStr)
				client.SetModelInfo(modelID, 0)
				*model = args
				fmt.Printf("Switched to model %s at %s\n", args, baseURLStr)
			}
			continue
		}

		if prompt == "/exit" || prompt == "/quit" {
			break
		}

		turnCtx, stopSignals := signal.NotifyContext(context.Background(), os.Interrupt)
		turnCtx, cancelTimeout := context.WithTimeout(turnCtx, *timeout)

		firstChunk := true
		_, err = bot.RunTurn(turnCtx, prompt, func(delta string) {
			if firstChunk {
				fmt.Println()
				firstChunk = false
			}
			fmt.Print(delta)
		})
		cancelTimeout()
		stopSignals()

		if err != nil {
			if errors.Is(err, context.Canceled) {
				fmt.Println("\n[Turn canceled]")
				continue
			}
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			continue
		}
	}
}
