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
)

const defaultSystemPrompt = `You are a concise coding agent editing a local repository in place.
Inspect files with list_files and read_file before modifying code.
Prefer edit_file for surgical changes to existing files and write_file for new files.
Verify changes with run_bash when appropriate.
No unicode characters or emojis in your responses. Use only ASCII characters.`

func main() {
	baseURL := flag.String("url", "http://localhost:8000/v1", "OpenAI-compatible vLLM base URL")
	rootDir := flag.String("root", ".", "Repository root directory to sandbox file operations")
	timeout := flag.Duration("timeout", 5*time.Minute, "Per-turn timeout")
	flag.Parse()

	registry, err := tools.NewRegistry(*rootDir, os.Stdin)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize tool registry: %v\n", err)
		os.Exit(1)
	}
	defer registry.Close()

	client := llm.NewClient(*baseURL, *timeout)
	bot := agent.New(client, registry, defaultSystemPrompt)

	checkCtx, cancelCheck := context.WithTimeout(context.Background(), 5*time.Second)
	initialModel, err := client.ActiveModel(checkCtx)
	cancelCheck()
	if err != nil {
		fmt.Printf("Warning: could not reach %s (%v)\n", *baseURL, err)
	} else {
		fmt.Printf("Connected to %s | Active model: %s | Sandbox: %s\n", *baseURL, initialModel, *rootDir)
	}
	fmt.Println("Type a prompt, /exit to quit, or press Ctrl+C during a turn to cancel.")

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
		if prompt == "/exit" || prompt == "/quit" {
			break
		}

		turnCtx, stopSignals := signal.NotifyContext(context.Background(), os.Interrupt)
		turnCtx, cancelTimeout := context.WithTimeout(turnCtx, *timeout)

		answer, err := bot.RunTurn(turnCtx, prompt)
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

		if answer != "" {
			fmt.Printf("\n%s\n", answer)
		}
	}
}
