//go:build windows || darwin

/*
 * webview_bindings.go - JavaScript Bindings für WebView
 * Enthält: Bind, Unbind, Dispatch und Callback-Funktionen
 *
 * MIT License - siehe webview_core.go für vollständige Lizenz
 */
package webview

/*
#include "webview.h"
#include <stdlib.h>
*/
import "C"

import (
	"encoding/json"
	"errors"
	"reflect"
	"unsafe"
)

func (w *webview) Dispatch(f func()) {
	m.Lock()
	for ; dispatch[index] != nil; index++ {
	}
	dispatch[index] = f
	m.Unlock()
	C.CgoWebViewDispatch(w.w, C.uintptr_t(index))
}

//export _webviewDispatchGoCallback
func _webviewDispatchGoCallback(index unsafe.Pointer) {
	m.Lock()
	f := dispatch[uintptr(index)]
	delete(dispatch, uintptr(index))
	m.Unlock()
	f()
}

//export _webviewBindingGoCallback
func _webviewBindingGoCallback(w C.webview_t, id *C.char, req *C.char, index uintptr) {
	m.Lock()
	f := bindings[index]
	m.Unlock()
	jsString := func(v interface{}) string { b, _ := json.Marshal(v); return string(b) }
	status := 0
	var result string
	if res, err := f(C.GoString(id), C.GoString(req)); err != nil {
		status = -1
		result = jsString(err.Error())
	} else if b, err := json.Marshal(res); err != nil {
		status = -1
		result = jsString(err.Error())
	} else {
		status = 0
		result = string(b)
	}
	s := C.CString(result)
	defer C.free(unsafe.Pointer(s))
	C.webview_return(w, id, C.int(status), s)
}

func (w *webview) Bind(name string, f interface{}) error {
	v := reflect.ValueOf(f)
	// f must be a function
	if v.Kind() != reflect.Func {
		return errors.New("only functions can be bound")
	}
	// f must return either value and error or just error
	if n := v.Type().NumOut(); n > 2 {
		return errors.New("function may only return a value or a value+error")
	}

	binding := func(id, req string) (interface{}, error) {
		raw := []json.RawMessage{}
		if err := json.Unmarshal([]byte(req), &raw); err != nil {
			return nil, err
		}

		isVariadic := v.Type().IsVariadic()
		numIn := v.Type().NumIn()
		if (isVariadic && len(raw) < numIn-1) || (!isVariadic && len(raw) != numIn) {
			return nil, errors.New("function arguments mismatch")
		}
		args := []reflect.Value{}
		for i := range raw {
			var arg reflect.Value
			if isVariadic && i >= numIn-1 {
				arg = reflect.New(v.Type().In(numIn - 1).Elem())
			} else {
				arg = reflect.New(v.Type().In(i))
			}
			if err := json.Unmarshal(raw[i], arg.Interface()); err != nil {
				return nil, err
			}
			args = append(args, arg.Elem())
		}
		errorType := reflect.TypeOf((*error)(nil)).Elem()
		res := v.Call(args)
		switch len(res) {
		case 0:
			// No results from the function, just return nil
			return nil, nil
		case 1:
			// One result may be a value, or an error
			if res[0].Type().Implements(errorType) {
				if res[0].Interface() != nil {
					return nil, res[0].Interface().(error)
				}
				return nil, nil
			}
			return res[0].Interface(), nil
		case 2:
			// Two results: first one is value, second is error
			if !res[1].Type().Implements(errorType) {
				return nil, errors.New("second return value must be an error")
			}
			if res[1].Interface() == nil {
				return res[0].Interface(), nil
			}
			return res[0].Interface(), res[1].Interface().(error)
		default:
			return nil, errors.New("unexpected number of return values")
		}
	}

	m.Lock()
	for ; bindings[index] != nil; index++ {
	}
	bindings[index] = binding
	m.Unlock()
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	C.CgoWebViewBind(w.w, cname, C.uintptr_t(index))
	return nil
}

func (w *webview) Unbind(name string) error {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	C.CgoWebViewUnbind(w.w, cname)
	return nil
}
