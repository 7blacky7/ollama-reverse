//go:build mlx

// mlx_debug.go - Debug und Diagnose Funktionen
//
// Enthaelt:
// - DebugArrays, DebugArraysVerbose

package mlx

import "fmt"

// DebugArrays prints summary info about all tracked arrays.
func DebugArrays() {
	var totalBytes int64
	var keptCount, unkeptCount int
	for _, a := range arrays {
		if a.kept {
			keptCount++
		} else {
			unkeptCount++
		}
		totalBytes += a.Nbytes()
	}
	fmt.Printf("[DEBUG] Arrays: %d kept, %d unkept, %.2f GB total\n",
		keptCount, unkeptCount, float64(totalBytes)/(1024*1024*1024))
}

// DebugArraysVerbose prints detailed info about all tracked arrays, sorted by size.
func DebugArraysVerbose(topN int) {
	type arrayInfo struct {
		shape []int32
		dtype Dtype
		bytes int64
		kept  bool
	}

	var infos []arrayInfo
	var totalBytes int64
	for _, a := range arrays {
		bytes := a.Nbytes()
		infos = append(infos, arrayInfo{
			shape: a.Shape(),
			dtype: a.Dtype(),
			bytes: bytes,
			kept:  a.kept,
		})
		totalBytes += bytes
	}

	// Sort by size descending
	for i := 0; i < len(infos)-1; i++ {
		for j := i + 1; j < len(infos); j++ {
			if infos[j].bytes > infos[i].bytes {
				infos[i], infos[j] = infos[j], infos[i]
			}
		}
	}

	fmt.Printf("[DEBUG] %d arrays, %.2f GB total:\n", len(infos), float64(totalBytes)/(1024*1024*1024))
	for i, info := range infos {
		if i >= topN {
			break
		}
		keptStr := ""
		if info.kept {
			keptStr = " [kept]"
		}
		fmt.Printf("  %3d. %8.2f MB  %v %v%s\n",
			i+1, float64(info.bytes)/(1024*1024), info.shape, info.dtype, keptStr)
	}
}
