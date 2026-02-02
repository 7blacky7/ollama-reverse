# LOC-Audit Report: convert/ und runner/
## Phase 11 - Compliance-Pruefung

**Datum:** 2026-02-02
**Regel:** Max 300 LOC pro Datei (Kommentare/Leerzeilen zaehlen nicht)
**Gepruefte Verzeichnisse:** `convert/`, `runner/`
**Anzahl Dateien:** 67

---

## ZUSAMMENFASSUNG

| Metrik | Wert |
|--------|------|
| Gepruefte Dateien | 67 |
| Dateien ueber 300 LOC | 6 |
| Compliance-Rate | 91.0% |
| Groesste Datei | sentencepiece_model.pb.go (1153 LOC) |

---

## VERSTOSS-TABELLE (Dateien > 300 LOC)

| Datei | LOC | Total Lines | Ueberschreitung | Empfehlung |
|-------|-----|-------------|-----------------|------------|
| convert/sentencepiece/sentencepiece_model.pb.go | 1153 | 1497 | +853 | AUSNAHME: Auto-generiert (protoc) |
| convert/tensor_test.go | 811 | 997 | +511 | Aufteilen nach Testgruppen |
| runner/ollamarunner/cache_test.go | 556 | 593 | +256 | Aufteilen nach Testgruppen |
| runner/common/logprob_test.go | 419 | 498 | +119 | Aufteilen nach Testgruppen |
| convert/convert_test.go | 399 | 471 | +99 | Aufteilen nach Modell-Tests |
| convert/tokenizer_test.go | 315 | 325 | +15 | Aufteilen nach Testgruppen |

---

## TOP 20 DATEIEN NACH LOC

| Rang | Datei | LOC | Status |
|------|-------|-----|--------|
| 1 | convert/sentencepiece/sentencepiece_model.pb.go | 1153 | AUSNAHME (generiert) |
| 2 | convert/tensor_test.go | 811 | VERSTOSS |
| 3 | runner/ollamarunner/cache_test.go | 556 | VERSTOSS |
| 4 | runner/common/logprob_test.go | 419 | VERSTOSS |
| 5 | convert/convert_test.go | 399 | VERSTOSS |
| 6 | convert/tokenizer_test.go | 315 | VERSTOSS |
| 7 | runner/llamarunner/cache_test.go | 283 | OK |
| 8 | convert/reader_test.go | 275 | OK |
| 9 | convert/convert_gptoss.go | 233 | OK |
| 10 | convert/convert_glm4moelite.go | 222 | OK |
| 11 | runner/ollamarunner/cache.go | 221 | OK |
| 12 | runner/ollamarunner/runner_batch.go | 214 | OK |
| 13 | runner/llamarunner/batch.go | 202 | OK |
| 14 | runner/ollamarunner/runner_handlers.go | 201 | OK |
| 15 | convert/convert_mistral.go | 191 | OK |
| 16 | runner/llamarunner/handlers.go | 191 | OK |
| 17 | convert/vocabulary.go | 190 | OK |
| 18 | runner/llamarunner/cache.go | 187 | OK |
| 19 | convert/convert_llama.go | 183 | OK |
| 20 | convert/reader_safetensors.go | 177 | OK |

---

## EMPFEHLUNGEN ZUR AUFTEILUNG

### 1. convert/tensor_test.go (811 LOC -> 3 Dateien)
```
tensor_split_test.go      - Tests fuer Tensor-Splitting
tensor_merge_test.go      - Tests fuer Tensor-Merging
tensor_pad_test.go        - Tests fuer Tensor-Padding
```

### 2. runner/ollamarunner/cache_test.go (556 LOC -> 3 Dateien)
```
cache_basic_test.go       - Grundlegende Cache-Tests
cache_slot_test.go        - Slot-Management-Tests
cache_sequence_test.go    - Sequenz-Verwaltungs-Tests
```

### 3. runner/common/logprob_test.go (419 LOC -> 2 Dateien)
```
logprob_basic_test.go     - Standard-Logprob-Tests
logprob_edge_test.go      - Edge-Cases und Randfaelle
```

### 4. convert/convert_test.go (399 LOC -> 3 Dateien)
```
convert_llama_test.go     - Llama-Modell-Tests
convert_gemma_test.go     - Gemma-Modell-Tests
convert_misc_test.go      - Sonstige Modell-Tests
```

### 5. convert/tokenizer_test.go (315 LOC -> 2 Dateien)
```
tokenizer_vocab_test.go   - Vokabular-Tests
tokenizer_special_test.go - Spezielle Token-Tests
```

---

## AUSNAHMEN

### sentencepiece_model.pb.go (1153 LOC)
- **Status:** AUSNAHME GENEHMIGT
- **Grund:** Auto-generierte Protobuf-Datei (protoc-gen-go v1.32.0)
- **Quelle:** sentencepiece_model.proto
- **Hinweis:** Nicht manuell bearbeiten, nur .proto aendern

---

## ALLE DATEIEN MIT LOC

### convert/ (42 Dateien)

| Datei | LOC | Total | Status |
|-------|-----|-------|--------|
| sentencepiece/sentencepiece_model.pb.go | 1153 | 1497 | AUSNAHME |
| tensor_test.go | 811 | 997 | VERSTOSS |
| convert_test.go | 399 | 471 | VERSTOSS |
| tokenizer_test.go | 315 | 325 | VERSTOSS |
| reader_test.go | 275 | 294 | OK |
| convert_gptoss.go | 233 | 269 | OK |
| convert_glm4moelite.go | 222 | 264 | OK |
| convert_mistral.go | 191 | 221 | OK |
| vocabulary.go | 190 | 242 | OK |
| convert_llama.go | 183 | 221 | OK |
| reader_safetensors.go | 177 | 208 | OK |
| convert_nomicbert.go | 167 | 213 | OK |
| convert_gemma3.go | 164 | 180 | OK |
| convert_mistral_causal.go | 154 | 181 | OK |
| convert_mllama.go | 150 | 179 | OK |
| convert_deepseek2.go | 150 | 173 | OK |
| convert_gemma3n.go | 149 | 165 | OK |
| convert_bert.go | 147 | 178 | OK |
| convert_llama4.go | 140 | 169 | OK |
| tokenizer_spm.go | 138 | 172 | OK |
| convert_llama_adapter.go | 135 | 170 | OK |
| convert_qwen3.go | 134 | 157 | OK |
| convert_model.go | 132 | 156 | OK |
| convert_types.go | 124 | 183 | OK |
| tensor.go | 122 | 155 | OK |
| convert_deepseekocr.go | 122 | 136 | OK |
| tokenizer_parser.go | 120 | 163 | OK |
| convert_phi3.go | 106 | 122 | OK |
| convert_olmo.go | 101 | 117 | OK |
| convert_qwen3vl.go | 98 | 116 | OK |
| convert_qwen25vl.go | 88 | 102 | OK |
| convert_gemma.go | 85 | 100 | OK |
| convert_lfm2.go | 83 | 100 | OK |
| reader.go | 80 | 95 | OK |
| convert_gemma2_adapter.go | 74 | 92 | OK |
| convert_qwen2.go | 72 | 81 | OK |
| convert_commandr.go | 67 | 76 | OK |
| convert_mixtral.go | 54 | 64 | OK |
| reader_torch.go | 48 | 59 | OK |
| convert_gemma2.go | 46 | 49 | OK |
| convert_adapter.go | 40 | 52 | OK |
| convert_kv.go | 37 | 57 | OK |

### runner/ (25 Dateien)

| Datei | LOC | Total | Status |
|-------|-----|-------|--------|
| ollamarunner/cache_test.go | 556 | 593 | VERSTOSS |
| common/logprob_test.go | 419 | 498 | VERSTOSS |
| llamarunner/cache_test.go | 283 | 292 | OK |
| ollamarunner/cache.go | 221 | 315 | OK |
| ollamarunner/runner_batch.go | 214 | 271 | OK |
| llamarunner/batch.go | 202 | 284 | OK |
| ollamarunner/runner_handlers.go | 201 | 238 | OK |
| llamarunner/handlers.go | 191 | 229 | OK |
| llamarunner/cache.go | 187 | 276 | OK |
| llamarunner/server.go | 166 | 217 | OK |
| ollamarunner/runner_model.go | 166 | 218 | OK |
| ollamarunner/runner_compute.go | 150 | 207 | OK |
| llamarunner/sequence.go | 143 | 195 | OK |
| ollamarunner/runner_sequence.go | 134 | 183 | OK |
| common/stop_test.go | 124 | 129 | OK |
| llamarunner/image.go | 114 | 149 | OK |
| ollamarunner/runner_load.go | 113 | 154 | OK |
| ollamarunner/runner_types.go | 84 | 194 | OK |
| llamarunner/types.go | 78 | 171 | OK |
| ollamarunner/multimodal.go | 75 | 116 | OK |
| common/stop.go | 68 | 96 | OK |
| ollamarunner/runner.go | 64 | 86 | OK |
| llamarunner/image_test.go | 64 | 82 | OK |
| common/logprob.go | 59 | 79 | OK |
| runner.go | 28 | 32 | OK |

---

## STATUS

```
+--------------------------------------------------------------+
|                    AUDIT-ERGEBNIS                            |
+--------------------------------------------------------------+
|  Status:           BEDINGT BESTANDEN                         |
|  Compliance:       91.0% (61/67 Dateien)                     |
|  Verstoesse:       5 (ohne Ausnahmen)                        |
|  Ausnahmen:        1 (generierte Protobuf-Datei)             |
|  Empfehlung:       Test-Dateien aufteilen                    |
+--------------------------------------------------------------+
```

**Wichtiger Hinweis:** Alle Verstoesse betreffen Test-Dateien (_test.go).
Produktiver Code ist vollstaendig compliant mit der 300 LOC Regel.

---

## NAECHSTE SCHRITTE

1. **Prioritaet HOCH:** convert/tensor_test.go aufteilen (811 LOC)
2. **Prioritaet HOCH:** runner/ollamarunner/cache_test.go aufteilen (556 LOC)
3. **Prioritaet MITTEL:** runner/common/logprob_test.go aufteilen (419 LOC)
4. **Prioritaet MITTEL:** convert/convert_test.go aufteilen (399 LOC)
5. **Prioritaet NIEDRIG:** convert/tokenizer_test.go aufteilen (315 LOC)

---

*Generiert von LOC-Audit Agent, Phase 11*
