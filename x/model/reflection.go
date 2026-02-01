// Package model - Reflection-Logik fuer Tensor-Mapping.
//
// Diese Datei enthaelt:
// - populateFields: Rekursive Feld-Population via Reflection
// - setPointer: Pointer/Interface-Handling
// - Tag: GGUF-Tag-Struktur fuer Tensor-Mapping
// - parseTag: Tag-Parsing-Logik
// - canNil: Typ-Pruefung fuer nil-faehige Typen
package model

import (
	"log/slog"
	"reflect"
	"strconv"
	"strings"

	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/ml"
)

// ============================================================================
// Tag-Struktur
// ============================================================================

// Tag repraesentiert einen geparsten GGUF-Tag fuer Tensor-Mapping.
type Tag struct {
	name string
	// prefix und suffix werden auf Child-Tags angewendet
	prefix       string
	suffix       string
	alternatives []string
}

// ============================================================================
// Reflection-Funktionen
// ============================================================================

// populateFields fuellt Struct-Felder rekursiv mit Tensoren aus dem Backend.
// Verwendet GGUF-Tags zur Zuordnung von Tensor-Namen zu Struct-Feldern.
func populateFields(base Base, v reflect.Value, tags ...Tag) reflect.Value {
	t := v.Type()

	if t.Kind() == reflect.Struct {
		allNil := true
		for i := range t.NumField() {
			tt := t.Field(i).Type
			vv := v.Field(i)
			if !vv.CanSet() {
				continue
			}

			// Kopie erstellen
			tagsCopy := tags
			if tag := t.Field(i).Tag.Get("gguf"); tag != "" {
				tagsCopy = append(tagsCopy, parseTag(tag))
			}

			if tt == reflect.TypeOf((*Base)(nil)).Elem() {
				vv.Set(reflect.ValueOf(base))
			} else if tt == reflect.TypeOf((*ml.Tensor)(nil)).Elem() {
				var fn func([]Tag, string, string) [][]string
				fn = func(tags []Tag, prefix, suffix string) (fullNames [][]string) {
					if len(tags) > 0 {
						var names []string
						if tags[0].name != "" {
							for _, n := range append([]string{tags[0].name}, tags[0].alternatives...) {
								names = append(names, prefix+n+suffix)
							}
						}
						childNames := fn(tags[1:], tags[0].prefix, tags[0].suffix)
						if len(names) == 0 {
							// Aktueller Tag hat keinen Namen, nur Child-Namen verwenden
							fullNames = append(fullNames, childNames...)
						} else if len(childNames) == 0 {
							// Aktueller Tag hat Namen aber keine Children, Branches fuer jeden Namen erstellen
							for _, name := range names {
								fullNames = append(fullNames, []string{name})
							}
						} else {
							// Jeden Namen mit jedem Child mergen
							for _, name := range names {
								for _, childName := range childNames {
									fullNames = append(fullNames, append([]string{name}, childName...))
								}
							}
						}
					}

					return fullNames
				}

				names := fn(tagsCopy, "", "")
				for _, name := range names {
					if tensor := base.Backend().Get(strings.Join(name, ".")); tensor != nil {
						logutil.Trace("found tensor", "", tensor)
						vv.Set(reflect.ValueOf(tensor))
						break
					}
				}
			} else if tt.Kind() == reflect.Pointer || tt.Kind() == reflect.Interface {
				setPointer(base, vv, tagsCopy)
			} else if tt.Kind() == reflect.Slice || tt.Kind() == reflect.Array {
				for i := range vv.Len() {
					vvv := vv.Index(i)
					if vvv.Kind() == reflect.Pointer || vvv.Kind() == reflect.Interface {
						setPointer(base, vvv, append(tagsCopy, Tag{name: strconv.Itoa(i)}))
					} else {
						vvv.Set(populateFields(base, vvv, append(tagsCopy, Tag{name: strconv.Itoa(i)})...))
					}
				}
			}

			if !canNil(tt) || !vv.IsNil() {
				allNil = false
			}
		}

		if allNil {
			return reflect.Zero(t)
		}
	}

	return v
}

// setPointer behandelt Pointer- und Interface-Felder.
func setPointer(base Base, v reflect.Value, tags []Tag) {
	vv := v
	if v.Kind() == reflect.Interface {
		if v.IsNil() {
			return
		}

		vv = vv.Elem()
	}

	vv = reflect.Indirect(vv)
	if v.IsNil() {
		vv = reflect.New(v.Type().Elem()).Elem()
	}

	if f := populateFields(base, vv, tags...); f.CanAddr() {
		v.Set(f.Addr())
	}
}

// parseTag parst einen GGUF-Tag-String in eine Tag-Struktur.
// Format: "name,alt:alternative,pre:prefix,suf:suffix"
func parseTag(s string) (tag Tag) {
	parts := strings.Split(s, ",")
	if len(parts) > 0 {
		tag.name = parts[0]

		for _, part := range parts[1:] {
			if value, ok := strings.CutPrefix(part, "alt:"); ok && tag.name == "" {
				// Alternative zum Primary erhoehen, wenn kein Primary gegeben
				tag.name = value
				slog.Warn("gguf tag has alt: but no primary name", "tag", s)
			} else if ok {
				tag.alternatives = append(tag.alternatives, value)
			}
			if value, ok := strings.CutPrefix(part, "pre:"); ok {
				tag.prefix = value
			}
			if value, ok := strings.CutPrefix(part, "suf:"); ok {
				tag.suffix = value
			}
		}
	}

	return
}

// canNil prueft, ob ein Typ nil sein kann.
func canNil(t reflect.Type) bool {
	return t.Kind() == reflect.Chan ||
		t.Kind() == reflect.Func ||
		t.Kind() == reflect.Interface ||
		t.Kind() == reflect.Map ||
		t.Kind() == reflect.Pointer ||
		t.Kind() == reflect.Slice
}
