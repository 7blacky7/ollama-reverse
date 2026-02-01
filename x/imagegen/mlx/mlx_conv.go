//go:build mlx

// mlx_conv.go - Convolution Operationen
//
// Enthaelt:
// - Conv1d, ConvTranspose1d, DepthwiseConv1d
// - Conv2d, Conv3d
// - Pad

package mlx

/*
#cgo CFLAGS: -O3 -I${SRCDIR}/../../../build/_deps/mlx-c-src -I${SRCDIR}
#cgo darwin LDFLAGS: -lc++ -framework Metal -framework Foundation -framework Accelerate
#cgo linux LDFLAGS: -lstdc++ -ldl
#cgo windows LDFLAGS: -lstdc++

#include "mlx.h"
#include <stdlib.h>

static mlx_stream _default_stream = {0};

static inline mlx_stream default_stream() {
    if (_default_stream.ctx == NULL) {
        _default_stream = mlx_default_gpu_stream_new();
    }
    return _default_stream;
}
*/
import "C"
import "unsafe"

// Conv1d performs 1D convolution
// x: [B, L, Cin], weight: [Cout, K, Cin] (MLX uses NLC layout)
// bias: optional (nil for no bias)
func Conv1d(x, weight *Array, bias *Array, stride int32) *Array {
	res := C.mlx_array_new()
	C.mlx_conv1d(&res, x.c, weight.c, C.int(stride), C.int(0), C.int(1), 1, C.default_stream())
	// Apply bias if provided
	if bias != nil {
		biased := C.mlx_array_new()
		C.mlx_add(&biased, res, bias.c, C.default_stream())
		C.mlx_array_free(res)
		return newArray(biased)
	}
	return newArray(res)
}

// ConvTranspose1d performs transposed 1D convolution
// x: [B, L, Cin], weight: [Cout, K, Cin] (MLX uses NLC layout)
// bias: optional (nil for no bias)
func ConvTranspose1d(x, weight *Array, bias *Array, stride int32) *Array {
	res := C.mlx_array_new()
	// stride, padding, dilation, output_padding, groups
	C.mlx_conv_transpose1d(&res, x.c, weight.c, C.int(stride), 0, 1, 0, 1, C.default_stream())
	// Apply bias if provided
	if bias != nil {
		biased := C.mlx_array_new()
		C.mlx_add(&biased, res, bias.c, C.default_stream())
		C.mlx_array_free(res)
		return newArray(biased)
	}
	return newArray(res)
}

// DepthwiseConv1d performs depthwise 1D convolution (groups=Cin)
// x: [B, L, C], weight: [1, K, C] (groups = C)
// bias: optional (nil for no bias)
func DepthwiseConv1d(x, weight *Array, bias *Array) *Array {
	// Get number of input channels for groups
	shape := x.Shape()
	groups := int(shape[len(shape)-1])
	res := C.mlx_array_new()
	C.mlx_conv1d(&res, x.c, weight.c, 1, 0, 1, C.int(groups), C.default_stream())
	// Apply bias if provided
	if bias != nil {
		biased := C.mlx_array_new()
		C.mlx_add(&biased, res, bias.c, C.default_stream())
		C.mlx_array_free(res)
		return newArray(biased)
	}
	return newArray(res)
}

// Conv2d performs 2D convolution
// input: [N, H, W, C], weight: [O, kH, kW, C]  (MLX uses NHWC layout)
// Returns: [N, H', W', O]
func Conv2d(input, weight *Array, stride, padding int32) *Array {
	res := C.mlx_array_new()
	C.mlx_conv2d(&res, input.c, weight.c, C.int(stride), C.int(stride), C.int(padding), C.int(padding), 1, 1, 1, C.default_stream())
	return newArray(res)
}

// Conv3d performs 3D convolution
// input: [N, D, H, W, C], weight: [O, kD, kH, kW, C]  (MLX uses NDHWC layout)
// Returns: [N, D', H', W', O]
func Conv3d(input, weight *Array, strideD, strideH, strideW, padD, padH, padW int32) *Array {
	res := C.mlx_array_new()
	C.mlx_conv3d(&res, input.c, weight.c, C.int(strideD), C.int(strideH), C.int(strideW), C.int(padD), C.int(padH), C.int(padW), 1, 1, 1, 1, C.default_stream())
	return newArray(res)
}

// Pad pads an array with zeros
// paddings: [before_0, after_0, before_1, after_1, ...] for each dimension
func Pad(a *Array, paddings []int32) *Array {
	numAxes := len(paddings) / 2
	// Convert to low/high pairs
	lowPad := make([]C.int, numAxes)
	highPad := make([]C.int, numAxes)
	for i := 0; i < numAxes; i++ {
		lowPad[i] = C.int(paddings[i*2])
		highPad[i] = C.int(paddings[i*2+1])
	}
	zero := C.mlx_array_new_float(0.0)
	res := C.mlx_array_new()
	// mlx_pad takes axes, low, high arrays
	axes := make([]C.int, numAxes)
	for i := 0; i < numAxes; i++ {
		axes[i] = C.int(i)
	}
	cMode := C.CString("constant")
	defer C.free(unsafe.Pointer(cMode))
	C.mlx_pad(&res, a.c, &axes[0], C.size_t(numAxes), &lowPad[0], C.size_t(numAxes), &highPad[0], C.size_t(numAxes), zero, cMode, C.default_stream())
	C.mlx_array_free(zero)
	return newArray(res)
}
