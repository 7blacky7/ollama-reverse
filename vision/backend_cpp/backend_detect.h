/**
 * MODUL: backend_detect.h
 * ZWECK: C-API fuer Backend-Erkennung (CUDA/Metal)
 * INPUT: Device-IDs
 * OUTPUT: Device-Informationen
 * NEBENEFFEKTE: Hardware-Abfragen
 * ABHAENGIGKEITEN: Keine (reine C-API)
 * HINWEISE: CGO-kompatibel durch C-Linkage, plattformunabhaengige Strukturen
 */

#ifndef BACKEND_DETECT_H
#define BACKEND_DETECT_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Konstanten
 * ============================================================================ */

/* Maximale Laenge fuer Geraete-Namen */
#define BACKEND_MAX_NAME_LEN 256

/* ============================================================================
 * Strukturen
 * ============================================================================ */

/**
 * Geraete-Informationen (plattformunabhaengig)
 * Wird von CUDA und Metal Funktionen befuellt
 */
typedef struct backend_device_info {
    char     name[BACKEND_MAX_NAME_LEN];  /* Lesbarer Geraetename */
    uint64_t memory_total;                 /* Gesamtspeicher in Bytes */
    uint64_t memory_free;                  /* Freier Speicher in Bytes */
    int32_t  compute_major;                /* Compute Capability Major (CUDA) */
    int32_t  compute_minor;                /* Compute Capability Minor (CUDA) */
    int32_t  device_id;                    /* Geraete-Index */
} backend_device_info;

/* ============================================================================
 * CUDA Funktionen
 * ============================================================================ */

/**
 * Prueft ob CUDA verfuegbar ist
 * @return true wenn CUDA Runtime geladen werden kann
 */
bool backend_cuda_available(void);

/**
 * Gibt die Anzahl der CUDA-Geraete zurueck
 * @return Anzahl GPUs, 0 wenn keine vorhanden
 */
int backend_cuda_device_count(void);

/**
 * Fragt Informationen fuer ein CUDA-Geraet ab
 * @param device_id GPU-Index (0-basiert)
 * @param info Output-Struktur (muss vom Aufrufer alloziert sein)
 * @return 0 bei Erfolg, Fehlercode sonst
 */
int backend_cuda_get_device(int device_id, backend_device_info* info);

/**
 * Setzt das aktive CUDA-Geraet
 * @param device_id GPU-Index
 * @return 0 bei Erfolg, Fehlercode sonst
 */
int backend_cuda_set_device(int device_id);

/**
 * Gibt aktuellen freien Speicher zurueck
 * @param device_id GPU-Index
 * @return Freier Speicher in Bytes
 */
uint64_t backend_cuda_get_free_memory(int device_id);

/* ============================================================================
 * Metal Funktionen
 * ============================================================================ */

/**
 * Prueft ob Metal verfuegbar ist
 * @return true auf macOS/iOS mit Metal-Unterstuetzung
 */
bool backend_metal_available(void);

/**
 * Gibt die Anzahl der Metal-Geraete zurueck
 * @return Anzahl GPUs (typischerweise 1 auf Apple Silicon)
 */
int backend_metal_device_count(void);

/**
 * Fragt Informationen fuer ein Metal-Geraet ab
 * @param device_id GPU-Index (normalerweise 0)
 * @param info Output-Struktur
 * @return 0 bei Erfolg, Fehlercode sonst
 */
int backend_metal_get_device(int device_id, backend_device_info* info);

/**
 * Gibt empfohlenes Working Set Limit zurueck
 * @param device_id GPU-Index
 * @return Empfohlenes Speicherlimit in Bytes
 */
uint64_t backend_metal_get_recommended_memory(int device_id);

/* ============================================================================
 * Utility Funktionen
 * ============================================================================ */

/**
 * Gibt den Namen des besten verfuegbaren Backends zurueck
 * Prioritaet: CUDA > Metal > CPU
 * @return "cuda", "metal" oder "cpu"
 */
const char* backend_get_best(void);

/**
 * Initialisiert alle Backends (optional)
 * Kann Performance verbessern durch Vorinitialisierung
 */
void backend_init_all(void);

#ifdef __cplusplus
}
#endif

#endif /* BACKEND_DETECT_H */
