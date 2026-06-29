// CurNodeFlags type alias for per-node flags at execution time.
//
// In release builds (!VERBOSE and no demo counters), node
// flags are never written during inference, so we alias CurNodeFlags to the
// const variant.  This lets the compiler catch accidental writes and avoids
// the NVM round-trip that commit_node_flags() would otherwise do.
//
// In debug or demo-counter builds the flags may be mutated (e.g., to record
// cumulative_jobs), so the alias resolves to a non-const struct and the
// shadow-copy write path is activated.

#pragma once

#include "config.h"

#if VERBOSE || ENABLE_DEMO_COUNTERS
typedef struct NodeFlags CurNodeFlags;
#else
typedef const struct NodeFlags CurNodeFlags;
#endif
