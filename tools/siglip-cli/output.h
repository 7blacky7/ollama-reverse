/**
 * output.h - Output-Formate fuer SigLIP CLI
 *
 * Dieses Modul stellt Funktionen bereit fuer:
 * - JSON Export
 * - Binary Export (little-endian float32)
 * - NumPy-kompatibles Format (.npy)
 */

#ifndef SIGLIP_CLI_OUTPUT_H
#define SIGLIP_CLI_OUTPUT_H

#include "../../siglip/siglip.h"
#include <string>
#include <vector>

// ============================================================================
// Output-Funktionen
// ============================================================================

/**
 * Schreibt Embedding als JSON
 *
 * @param emb      Embedding-Daten
 * @param path     Output-Pfad (leer = stdout)
 * @param filename Optional: Dateiname fuer Batch-Modus
 * @return         true bei Erfolg
 */
bool write_json(const siglip_embedding * emb, const std::string & path,
                const std::string & filename = "");

/**
 * Schreibt Batch-Embeddings als JSON-Array
 *
 * @param embeddings  Vektor von Embeddings
 * @param filenames   Zugehoerige Dateinamen
 * @param path        Output-Pfad
 * @return            true bei Erfolg
 */
bool write_json_batch(const std::vector<siglip_embedding *> & embeddings,
                      const std::vector<std::string> & filenames,
                      const std::string & path);

/**
 * Schreibt Embedding als Binary (little-endian float32)
 *
 * @param emb   Embedding-Daten
 * @param path  Output-Pfad (erforderlich)
 * @return      true bei Erfolg
 */
bool write_binary(const siglip_embedding * emb, const std::string & path);

/**
 * Schreibt Batch-Embeddings als Binary
 *
 * @param embeddings  Vektor von Embeddings
 * @param path        Output-Pfad
 * @return            true bei Erfolg
 */
bool write_binary_batch(const std::vector<siglip_embedding *> & embeddings,
                        const std::string & path);

/**
 * Schreibt Embedding im NumPy .npy Format
 *
 * @param emb   Embedding-Daten
 * @param path  Output-Pfad (erforderlich)
 * @return      true bei Erfolg
 */
bool write_numpy(const siglip_embedding * emb, const std::string & path);

/**
 * Schreibt Batch-Embeddings als NumPy 2D-Array
 *
 * @param embeddings  Vektor von Embeddings
 * @param path        Output-Pfad
 * @return            true bei Erfolg
 */
bool write_numpy_batch(const std::vector<siglip_embedding *> & embeddings,
                       const std::string & path);

#endif // SIGLIP_CLI_OUTPUT_H
