//go:build vision && !siglip

// MODUL: vision_adapter_stub
// ZWECK: Stub fuer siglip-spezifische Funktionen wenn siglip-Tag nicht gesetzt ist
// INPUT: Keine
// OUTPUT: Stub-Implementierung die auf vision.Registry verweist
// NEBENEFFEKTE: Keine
// ABHAENGIGKEITEN: vision (VisionEncoder)
// HINWEISE: Wird nur mit -tags "vision" (ohne siglip) kompiliert
//           getModel ist in router_vision.go definiert und verwendet vision.Registry

package server

// Keine zusaetzlichen Definitionen noetig - router_vision.go verwendet vision.Registry
