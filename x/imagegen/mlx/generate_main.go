//go:build ignore

// generate_main.go - Hauptprogramm fuer MLX-Wrapper-Generator
//
// Usage: go run generate_types.go generate_parser.go generate_main.go <mlx-c-include-dir> <output-header> [output-impl]
//
// Enthaelt:
// - main: Einstiegspunkt und Kommandozeilenverarbeitung
// - Header-Parsing und Wrapper-Generierung
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
)

func main() {
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage: go run generate_wrappers.go <mlx-c-include-dir> <output-header> [output-impl]\n")
		fmt.Fprintf(flag.CommandLine.Output(), "Generate MLX-C dynamic loading wrappers.\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	args := flag.Args()
	if len(args) < 2 {
		fmt.Fprintf(flag.CommandLine.Output(), "ERROR: Missing required arguments\n\n")
		flag.Usage()
		os.Exit(1)
	}

	headerDir := args[0]
	outputHeader := args[1]
	// Default implementation file is same name with .c extension
	outputImpl := outputHeader
	if len(args) > 2 {
		outputImpl = args[2]
	} else if strings.HasSuffix(outputHeader, ".h") {
		outputImpl = outputHeader[:len(outputHeader)-2] + ".c"
	}

	// Check if header directory exists
	if _, err := os.Stat(headerDir); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "ERROR: MLX-C headers directory not found at: %s\n\n", headerDir)
		fmt.Fprintf(os.Stderr, "Please run CMake first to download MLX-C dependencies:\n")
		fmt.Fprintf(os.Stderr, "  cmake -B build\n\n")
		fmt.Fprintf(os.Stderr, "The CMake build will download and extract MLX-C headers needed for wrapper generation.\n")
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "Parsing MLX-C headers from: %s\n", headerDir)

	// Find all headers
	headers, err := findHeaders(headerDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: Failed to find header files: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Found %d header files\n", len(headers))

	// Parse all headers
	var allFunctions []Function
	seen := make(map[string]bool)

	for _, header := range headers {
		content, err := os.ReadFile(header)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading %s: %v\n", header, err)
			continue
		}

		cleaned := cleanContent(string(content))
		functions := parseFunctions(cleaned)

		// Deduplicate
		for _, fn := range functions {
			if !seen[fn.Name] {
				seen[fn.Name] = true
				allFunctions = append(allFunctions, fn)
			}
		}
	}

	fmt.Fprintf(os.Stderr, "Found %d unique function declarations\n", len(allFunctions))

	// Generate wrapper files
	if err := generateWrapperFiles(allFunctions, outputHeader, outputImpl); err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: Failed to generate wrapper files: %v\n", err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "Generated %s and %s successfully\n", outputHeader, outputImpl)
}
