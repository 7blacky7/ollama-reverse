// Package model - Reflection-basierte Tensor-Population
//
// Dieses Modul enthält die Reflection-Logik zum automatischen Befüllen
// von Model-Strukturen mit Tensoren aus dem Backend.
//
// Hauptkomponenten:
// - populateFields: Befüllt Strukturfelder rekursiv mit Tensoren
// - setPointer: Setzt Pointer-Felder in Strukturen
// - Tag: GGUF-Tag-Struktur für Tensor-Namen
// - parseTag: Parst GGUF-Tags aus Struct-Tags

package model

import (
	"log/slog"
	"reflect"
	"strconv"
	"strings"

	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

// Tag repräsentiert einen geparseten GGUF-Tag
type Tag struct {
	name,
	// prefix und suffix werden auf Kind-Tags angewendet
	prefix,
	suffix string
	alternatives []string
}

// parseTag parst einen GGUF-Tag-String in eine Tag-Struktur
func parseTag(s string) (tag Tag) {
	parts := strings.Split(s, ",")
	if len(parts) > 0 {
		tag.name = parts[0]

		for _, part := range parts[1:] {
			if value, ok := strings.CutPrefix(part, "alt:"); ok && tag.name == "" {
				// Alternative zum Primärnamen erheben wenn kein Primärname
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

// canNil prüft ob ein Typ nil sein kann
func canNil(t reflect.Type) bool {
	return t.Kind() == reflect.Chan ||
		t.Kind() == reflect.Func ||
		t.Kind() == reflect.Interface ||
		t.Kind() == reflect.Map ||
		t.Kind() == reflect.Pointer ||
		t.Kind() == reflect.Slice
}

// populateFields befüllt Strukturfelder rekursiv mit Tensoren aus dem Backend
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
				names := buildTensorNames(tagsCopy, "", "")
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

// buildTensorNames baut die vollständigen Tensor-Namen aus Tags
func buildTensorNames(tags []Tag, prefix, suffix string) (fullNames [][]string) {
	if len(tags) > 0 {
		var names []string
		if tags[0].name != "" {
			for _, n := range append([]string{tags[0].name}, tags[0].alternatives...) {
				names = append(names, prefix+n+suffix)
			}
		}
		childNames := buildTensorNames(tags[1:], tags[0].prefix, tags[0].suffix)
		if len(names) == 0 {
			// Aktueller Tag hat keinen Namen, nur Kind-Namen verwenden
			fullNames = append(fullNames, childNames...)
		} else if len(childNames) == 0 {
			// Aktueller Tag hat Namen aber keine Kinder, Branches für jeden Namen erstellen
			for _, name := range names {
				fullNames = append(fullNames, []string{name})
			}
		} else {
			// Jeden Namen mit jedem Kind zusammenführen
			for _, name := range names {
				for _, childName := range childNames {
					fullNames = append(fullNames, append([]string{name}, childName...))
				}
			}
		}
	}

	return fullNames
}

// setPointer setzt Pointer-Felder in Strukturen
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
