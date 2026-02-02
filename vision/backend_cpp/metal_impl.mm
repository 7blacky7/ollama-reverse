/**
 * MODUL: metal_impl.mm
 * ZWECK: Metal Framework Wrapper fuer Device-Erkennung
 * INPUT: Device-IDs
 * OUTPUT: Device-Name, Speicher-Informationen
 * NEBENEFFEKTE: Metal Framework Initialisierung
 * ABHAENGIGKEITEN: Metal.framework, Foundation.framework
 * HINWEISE: Nur auf macOS/iOS kompilierbar, Objective-C++ erforderlich
 */

#if defined(__APPLE__)

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstring>

extern "C" {

/* ============================================================================
 * Metal Verfuegbarkeit
 * ============================================================================ */

bool metal_is_available(void) {
    @autoreleasepool {
        /* Versuche Default-Device zu erhalten */
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return (device != nil);
    }
}

/* ============================================================================
 * Device-Zaehlung
 * ============================================================================ */

int metal_get_device_count(void) {
    @autoreleasepool {
#if TARGET_OS_OSX
        /* macOS: Kann mehrere GPUs haben (z.B. mit eGPU) */
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        int count = (int)[devices count];
        return count;
#else
        /* iOS/iPadOS: Immer genau ein Geraet */
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return (device != nil) ? 1 : 0;
#endif
    }
}

/* ============================================================================
 * Device-Informationen
 * ============================================================================ */

int metal_get_device_info(int device_id, char* name, size_t name_len,
                          uint64_t* memory_total, uint64_t* memory_free) {
    @autoreleasepool {
        id<MTLDevice> device = nil;

#if TARGET_OS_OSX
        /* macOS: Geraet nach Index waehlen */
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (device_id < 0 || device_id >= (int)[devices count]) {
            return -1;  /* Ungueltiger Index */
        }
        device = [devices objectAtIndex:device_id];
#else
        /* iOS: Nur Default-Device verfuegbar */
        if (device_id != 0) return -1;
        device = MTLCreateSystemDefaultDevice();
#endif

        if (device == nil) return -2;

        /* Name kopieren */
        if (name && name_len > 0) {
            const char* device_name = [[device name] UTF8String];
            strncpy(name, device_name, name_len - 1);
            name[name_len - 1] = '\0';
        }

        /* Speicher-Informationen */
        if (memory_total) {
            /* recommendedMaxWorkingSetSize ist der empfohlene Speicher */
            /* Auf Apple Silicon ist dies Unified Memory */
            *memory_total = [device recommendedMaxWorkingSetSize];
        }

        if (memory_free) {
            /* Metal bietet keine direkte Free-Memory Abfrage */
            /* Nutze recommendedMaxWorkingSetSize als Proxy */
            *memory_free = [device recommendedMaxWorkingSetSize];
        }

        return 0;
    }
}

/* ============================================================================
 * Empfohlenes Working Set
 * ============================================================================ */

uint64_t metal_get_recommended_working_set(int device_id) {
    @autoreleasepool {
        id<MTLDevice> device = nil;

#if TARGET_OS_OSX
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (device_id >= 0 && device_id < (int)[devices count]) {
            device = [devices objectAtIndex:device_id];
        }
#else
        if (device_id == 0) {
            device = MTLCreateSystemDefaultDevice();
        }
#endif

        if (device == nil) return 0;

        return [device recommendedMaxWorkingSetSize];
    }
}

/* ============================================================================
 * Zusaetzliche Metal-Infos (fuer Debugging)
 * ============================================================================ */

bool metal_device_is_low_power(int device_id) {
    @autoreleasepool {
#if TARGET_OS_OSX
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (device_id >= 0 && device_id < (int)[devices count]) {
            id<MTLDevice> device = [devices objectAtIndex:device_id];
            return [device isLowPower];
        }
#endif
        return false;
    }
}

bool metal_device_is_headless(int device_id) {
    @autoreleasepool {
#if TARGET_OS_OSX
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (device_id >= 0 && device_id < (int)[devices count]) {
            id<MTLDevice> device = [devices objectAtIndex:device_id];
            return [device isHeadless];
        }
#endif
        return false;
    }
}

} /* extern "C" */

#endif /* __APPLE__ */
