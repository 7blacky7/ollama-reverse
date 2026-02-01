// MODUL: encoders_nocgo
// ZWECK: Fallback fuer Builds ohne CGO-Support
// INPUT: Keine
// OUTPUT: Keine
// NEBENEFFEKTE: Gibt Warnung aus wenn kein Encoder verfuegbar
// ABHAENGIGKEITEN: Keine
// HINWEISE: Aktiv auf Windows oder wenn CGO deaktiviert

//go:build !unix || !cgo

package main

import "fmt"

func init() {
	// Hinweis fuer Builds ohne CGO
	// Encoder muessen manuell registriert werden oder via Plugin geladen
	_ = fmt.Sprintf // Verhindert unused import Fehler
}
