// types.go - Core API Types (Basis-Typen, Errors, Options, Metriken)
// Enthaelt: StatusError, AuthorizationError, ImageData, Metrics, Options, Runner, Duration, FormatParams
package api

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
)

// StatusError is an error with an HTTP status code and message.
type StatusError struct {
	StatusCode   int
	Status       string
	ErrorMessage string `json:"error"`
}

func (e StatusError) Error() string {
	switch {
	case e.Status != "" && e.ErrorMessage != "":
		return fmt.Sprintf("%s: %s", e.Status, e.ErrorMessage)
	case e.Status != "":
		return e.Status
	case e.ErrorMessage != "":
		return e.ErrorMessage
	default:
		// this should not happen
		return "something went wrong, please see the ollama server logs for details"
	}
}

type AuthorizationError struct {
	StatusCode int
	Status     string
	SigninURL  string `json:"signin_url"`
}

func (e AuthorizationError) Error() string {
	if e.Status != "" {
		return e.Status
	}
	return "something went wrong, please see the ollama server logs for details"
}

// ImageData represents the raw binary data of an image file.
type ImageData []byte

// Metrics enthaelt Performance-Metriken fuer Anfragen
type Metrics struct {
	TotalDuration      time.Duration `json:"total_duration,omitempty"`
	LoadDuration       time.Duration `json:"load_duration,omitempty"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       time.Duration `json:"eval_duration,omitempty"`
}

func (m *Metrics) Summary() {
	if m.TotalDuration > 0 {
		fmt.Fprintf(os.Stderr, "total duration:       %v\n", m.TotalDuration)
	}

	if m.LoadDuration > 0 {
		fmt.Fprintf(os.Stderr, "load duration:        %v\n", m.LoadDuration)
	}

	if m.PromptEvalCount > 0 {
		fmt.Fprintf(os.Stderr, "prompt eval count:    %d token(s)\n", m.PromptEvalCount)
	}

	if m.PromptEvalDuration > 0 {
		fmt.Fprintf(os.Stderr, "prompt eval duration: %s\n", m.PromptEvalDuration)
		fmt.Fprintf(os.Stderr, "prompt eval rate:     %.2f tokens/s\n", float64(m.PromptEvalCount)/m.PromptEvalDuration.Seconds())
	}

	if m.EvalCount > 0 {
		fmt.Fprintf(os.Stderr, "eval count:           %d token(s)\n", m.EvalCount)
	}

	if m.EvalDuration > 0 {
		fmt.Fprintf(os.Stderr, "eval duration:        %s\n", m.EvalDuration)
		fmt.Fprintf(os.Stderr, "eval rate:            %.2f tokens/s\n", float64(m.EvalCount)/m.EvalDuration.Seconds())
	}
}

// Options specified in [GenerateRequest]. If you add a new option here, also
// add it to the API docs.
type Options struct {
	Runner

	// Predict options used at runtime
	NumKeep          int      `json:"num_keep,omitempty"`
	Seed             int      `json:"seed,omitempty"`
	NumPredict       int      `json:"num_predict,omitempty"`
	TopK             int      `json:"top_k,omitempty"`
	TopP             float32  `json:"top_p,omitempty"`
	MinP             float32  `json:"min_p,omitempty"`
	TypicalP         float32  `json:"typical_p,omitempty"`
	RepeatLastN      int      `json:"repeat_last_n,omitempty"`
	Temperature      float32  `json:"temperature,omitempty"`
	RepeatPenalty    float32  `json:"repeat_penalty,omitempty"`
	PresencePenalty  float32  `json:"presence_penalty,omitempty"`
	FrequencyPenalty float32  `json:"frequency_penalty,omitempty"`
	Stop             []string `json:"stop,omitempty"`
}

// Runner options which must be set when the model is loaded into memory
type Runner struct {
	NumCtx    int   `json:"num_ctx,omitempty"`
	NumBatch  int   `json:"num_batch,omitempty"`
	NumGPU    int   `json:"num_gpu,omitempty"`
	MainGPU   int   `json:"main_gpu,omitempty"`
	UseMMap   *bool `json:"use_mmap,omitempty"`
	NumThread int   `json:"num_thread,omitempty"`
}

// DefaultOptions is the default set of options for [GenerateRequest]; these
// values are used unless the user specifies other values explicitly.
func DefaultOptions() Options {
	return Options{
		// options set on request to runner
		NumPredict: -1,

		// set a minimal num_keep to avoid issues on context shifts
		NumKeep:          4,
		Temperature:      0.8,
		TopK:             40,
		TopP:             0.9,
		TypicalP:         1.0,
		RepeatLastN:      64,
		RepeatPenalty:    1.1,
		PresencePenalty:  0.0,
		FrequencyPenalty: 0.0,
		Seed:             -1,

		Runner: Runner{
			// options set when the model is loaded
			NumCtx:    int(envconfig.ContextLength()),
			NumBatch:  512,
			NumGPU:    -1, // -1 here indicates that NumGPU should be set dynamically
			NumThread: 0,  // let the runtime decide
			UseMMap:   nil,
		},
	}
}

func (opts *Options) FromMap(m map[string]any) error {
	valueOpts := reflect.ValueOf(opts).Elem() // names of the fields in the options struct
	typeOpts := reflect.TypeOf(opts).Elem()   // types of the fields in the options struct

	// build map of json struct tags to their types
	jsonOpts := make(map[string]reflect.StructField)
	for _, field := range reflect.VisibleFields(typeOpts) {
		jsonTag := strings.Split(field.Tag.Get("json"), ",")[0]
		if jsonTag != "" {
			jsonOpts[jsonTag] = field
		}
	}

	for key, val := range m {
		opt, ok := jsonOpts[key]
		if !ok {
			slog.Warn("invalid option provided", "option", key)
			continue
		}

		field := valueOpts.FieldByName(opt.Name)
		if field.IsValid() && field.CanSet() {
			if val == nil {
				continue
			}

			switch field.Kind() {
			case reflect.Int:
				switch t := val.(type) {
				case int64:
					field.SetInt(t)
				case float64:
					// when JSON unmarshals numbers, it uses float64, not int
					field.SetInt(int64(t))
				default:
					return fmt.Errorf("option %q must be of type integer", key)
				}
			case reflect.Bool:
				val, ok := val.(bool)
				if !ok {
					return fmt.Errorf("option %q must be of type boolean", key)
				}
				field.SetBool(val)
			case reflect.Float32:
				// JSON unmarshals to float64
				val, ok := val.(float64)
				if !ok {
					return fmt.Errorf("option %q must be of type float32", key)
				}
				field.SetFloat(val)
			case reflect.String:
				val, ok := val.(string)
				if !ok {
					return fmt.Errorf("option %q must be of type string", key)
				}
				field.SetString(val)
			case reflect.Slice:
				// JSON unmarshals to []any, not []string
				val, ok := val.([]any)
				if !ok {
					return fmt.Errorf("option %q must be of type array", key)
				}
				// convert []any to []string
				slice := make([]string, len(val))
				for i, item := range val {
					str, ok := item.(string)
					if !ok {
						return fmt.Errorf("option %q must be of an array of strings", key)
					}
					slice[i] = str
				}
				field.Set(reflect.ValueOf(slice))
			case reflect.Pointer:
				var b bool
				if field.Type() == reflect.TypeOf(&b) {
					val, ok := val.(bool)
					if !ok {
						return fmt.Errorf("option %q must be of type boolean", key)
					}
					field.Set(reflect.ValueOf(&val))
				} else {
					return fmt.Errorf("unknown type loading config params: %v %v", field.Kind(), field.Type())
				}
			default:
				return fmt.Errorf("unknown type loading config params: %v", field.Kind())
			}
		}
	}

	return nil
}

// Duration ist ein JSON-serialisierbarer time.Duration Wrapper
type Duration struct {
	time.Duration
}

func (d Duration) MarshalJSON() ([]byte, error) {
	if d.Duration < 0 {
		return []byte("-1"), nil
	}
	return []byte("\"" + d.Duration.String() + "\""), nil
}

func (d *Duration) UnmarshalJSON(b []byte) (err error) {
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}

	d.Duration = 5 * time.Minute

	switch t := v.(type) {
	case float64:
		if t < 0 {
			d.Duration = time.Duration(math.MaxInt64)
		} else {
			d.Duration = time.Duration(t * float64(time.Second))
		}
	case string:
		d.Duration, err = time.ParseDuration(t)
		if err != nil {
			return err
		}
		if d.Duration < 0 {
			d.Duration = time.Duration(math.MaxInt64)
		}
	default:
		return fmt.Errorf("Unsupported type: '%s'", reflect.TypeOf(v))
	}

	return nil
}

// FormatParams converts specified parameter options to their correct types
func FormatParams(params map[string][]string) (map[string]any, error) {
	opts := Options{}
	valueOpts := reflect.ValueOf(&opts).Elem() // names of the fields in the options struct
	typeOpts := reflect.TypeOf(opts)           // types of the fields in the options struct

	// build map of json struct tags to their types
	jsonOpts := make(map[string]reflect.StructField)
	for _, field := range reflect.VisibleFields(typeOpts) {
		jsonTag := strings.Split(field.Tag.Get("json"), ",")[0]
		if jsonTag != "" {
			jsonOpts[jsonTag] = field
		}
	}

	out := make(map[string]any)
	// iterate params and set values based on json struct tags
	for key, vals := range params {
		if opt, ok := jsonOpts[key]; !ok {
			return nil, fmt.Errorf("unknown parameter '%s'", key)
		} else {
			field := valueOpts.FieldByName(opt.Name)
			if field.IsValid() && field.CanSet() {
				switch field.Kind() {
				case reflect.Float32:
					floatVal, err := strconv.ParseFloat(vals[0], 32)
					if err != nil {
						return nil, fmt.Errorf("invalid float value %s", vals)
					}

					out[key] = float32(floatVal)
				case reflect.Int:
					intVal, err := strconv.ParseInt(vals[0], 10, 64)
					if err != nil {
						return nil, fmt.Errorf("invalid int value %s", vals)
					}

					out[key] = intVal
				case reflect.Bool:
					boolVal, err := strconv.ParseBool(vals[0])
					if err != nil {
						return nil, fmt.Errorf("invalid bool value %s", vals)
					}

					out[key] = boolVal
				case reflect.String:
					out[key] = vals[0]
				case reflect.Slice:
					// TODO: only string slices are supported right now
					out[key] = vals
				case reflect.Pointer:
					var b bool
					if field.Type() == reflect.TypeOf(&b) {
						boolVal, err := strconv.ParseBool(vals[0])
						if err != nil {
							return nil, fmt.Errorf("invalid bool value %s", vals)
						}
						out[key] = &boolVal
					} else {
						return nil, fmt.Errorf("unknown type %s for %s", field.Kind(), key)
					}
				default:
					return nil, fmt.Errorf("unknown type %s for %s", field.Kind(), key)
				}
			}
		}
	}

	return out, nil
}
