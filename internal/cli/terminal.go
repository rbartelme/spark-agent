package cli

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"golang.org/x/term"
)

type PromptReader struct {
	fd      int
	isTTY   bool
	term    *term.Terminal
	scanner *bufio.Scanner
}

func NewPromptReader() *PromptReader {
	fd := int(os.Stdin.Fd())
	if !term.IsTerminal(fd) {
		return &PromptReader{
			fd:      fd,
			isTTY:   false,
			scanner: bufio.NewScanner(os.Stdin),
		}
	}

	rw := struct {
		io.Reader
		io.Writer
	}{os.Stdin, os.Stdout}

	t := term.NewTerminal(rw, "> ")
	t.AutoCompleteCallback = autocompleteSlashCommands

	return &PromptReader{
		fd:    fd,
		isTTY: true,
		term:  t,
	}
}

// ReadPrompt reads a single- or multi-line user prompt.
// Multi-line mode activates on trailing '\', unclosed ``` fences, or """ blocks.
func (p *PromptReader) ReadPrompt() (string, error) {
	if !p.isTTY {
		if !p.scanner.Scan() {
			if err := p.scanner.Err(); err != nil {
				return "", err
			}
			return "", io.EOF
		}
		return p.scanner.Text(), nil
	}

	oldState, err := term.MakeRaw(p.fd)
	if err != nil {
		return "", fmt.Errorf("enable raw mode: %w", err)
	}
	defer term.Restore(p.fd, oldState)

	if w, h, err := term.GetSize(p.fd); err == nil && w > 0 && h > 0 {
		_ = p.term.SetSize(w, h)
	}

	var lines []string
	inCodeFence := false
	inTripleQuote := false

	for {
		if len(lines) == 0 {
			p.term.SetPrompt("> ")
		} else {
			p.term.SetPrompt("... ")
		}

		line, err := p.term.ReadLine()
		if err != nil {
			// Ctrl+C in x/term returns io.EOF; if mid-multiline, cancel the draft instead of exiting.
			if errors.Is(err, io.EOF) && len(lines) > 0 {
				_, _ = p.term.Write([]byte("^C\n"))
				return "", nil
			}
			return "", err
		}

		if strings.Count(line, "```")%2 == 1 {
			inCodeFence = !inCodeFence
		}
		if strings.Count(line, `"""`)%2 == 1 {
			inTripleQuote = !inTripleQuote
		}

		if strings.HasSuffix(line, `\`) {
			lines = append(lines, strings.TrimSuffix(line, `\`))
			continue
		}

		lines = append(lines, line)

		if inCodeFence || inTripleQuote {
			continue
		}

		return strings.Join(lines, "\n"), nil
	}
}

func autocompleteSlashCommands(line string, pos int, key rune) (string, int, bool) {
	if key != '\t' || pos != len(line) {
		return "", 0, false
	}
	for _, cmd := range []string{"/exit", "/quit", "/model", "/model up", "/model down", "/model status", "/model list"} {
		if strings.HasPrefix(cmd, line) && line != "" {
			return cmd, len(cmd), true
		}
	}
	return "", 0, false
}
