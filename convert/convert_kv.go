// convert_kv.go - KV-Getter: Typisierte Zugriffsmethoden fuer KV-Map
// Hauptfunktionen: String, Uint, Float, Bool, Strings, Ints, Uints, Floats, Bools
package convert

// String - Gibt String-Wert zurueck
func (kv KV) String(key string, defaultValue ...string) string {
	val, _ := keyValue(kv, key, append(defaultValue, "")...)
	return val
}

// Uint - Gibt uint32-Wert zurueck
func (kv KV) Uint(key string, defaultValue ...uint32) uint32 {
	val, _ := keyValue(kv, key, append(defaultValue, 0)...)
	return val
}

// Float - Gibt float32-Wert zurueck
func (kv KV) Float(key string, defaultValue ...float32) float32 {
	val, _ := keyValue(kv, key, append(defaultValue, 0)...)
	return val
}

// Bool - Gibt bool-Wert zurueck
func (kv KV) Bool(key string, defaultValue ...bool) bool {
	val, _ := keyValue(kv, key, append(defaultValue, false)...)
	return val
}

// Strings - Gibt String-Array zurueck
func (kv KV) Strings(key string, defaultValue ...[]string) []string {
	val, _ := keyValue(kv, key, append(defaultValue, []string{""})...)
	return val
}

// Ints - Gibt int32-Array zurueck
func (kv KV) Ints(key string, defaultValue ...[]int32) []int32 {
	val, _ := keyValue(kv, key, append(defaultValue, []int32{0})...)
	return val
}

// Uints - Gibt uint32-Array zurueck
func (kv KV) Uints(key string, defaultValue ...[]uint32) []uint32 {
	val, _ := keyValue(kv, key, append(defaultValue, []uint32{0})...)
	return val
}

// Floats - Gibt float32-Array zurueck
func (kv KV) Floats(key string, defaultValue ...[]float32) []float32 {
	val, _ := keyValue(kv, key, append(defaultValue, []float32{0})...)
	return val
}

// Bools - Gibt bool-Array zurueck
func (kv KV) Bools(key string, defaultValue ...[]bool) []bool {
	val, _ := keyValue(kv, key, append(defaultValue, []bool{false})...)
	return val
}
