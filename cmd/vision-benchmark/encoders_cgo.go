// MODUL: encoders_cgo
// ZWECK: CGO-basierte Encoder-Registrierung fuer Unix-Builds
// INPUT: Keine
// OUTPUT: Keine (Seiteneffekt: Encoder-Registrierung)
// NEBENEFFEKTE: Registriert alle Vision-Encoder via init()
// ABHAENGIGKEITEN: vision/clip, vision/evaclip, vision/nomic, vision/openclip
// HINWEISE: Nur auf Unix mit CGO-Support, Windows nutzt encoders_nocgo.go

//go:build unix && cgo

package main

import (
	// Encoder-Registrierung via init()
	_ "github.com/ollama/ollama/vision/clip"
	_ "github.com/ollama/ollama/vision/evaclip"
	_ "github.com/ollama/ollama/vision/nomic"
	_ "github.com/ollama/ollama/vision/openclip"
)
