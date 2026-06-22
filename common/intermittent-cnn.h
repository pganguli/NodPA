// Power-failure recovery helpers for intermittent inference.
//
// RECOVERY MODEL
// ==============
// Each layer breaks its computation into "jobs", where one job produces
// BATCH_SIZE output values.  After each job the result is written to NVM.
// On power-on the system reads back how many jobs completed (via a footprint
// for HAWAII, or by scanning NVM for the last written batch otherwise), and
// resumes from that point.
//
// job_index_to_offset — translates a job index back to a flat NVM offset
//   in the output tensor.  The mapping is op-specific: Conv output is stored
//   in NWHC order (transposed) to make channel-wise tiling contiguous, while
//   other ops use a linear layout.
//
// batch_start — given the offset of the LAST element of a batch
//   (the value written to NVM as the footprint), returns the offset of the
//   FIRST element of that batch.  BATCH_SIZE-1 elements precede the sentinel.
//
// run_recovery — reads the footprint for the current layer (HAWAII variant)
//   or scans for the last completed job (non-HAWAII), and returns the index
//   of the first incomplete job so that execution can skip already-done work.

#pragma once

#include <cstdint>

#include "cnn_common.h"
#include "data.h"
#include "my_debug.h"

struct ParameterInfo;
struct Model;

uint32_t job_index_to_offset(const ParameterInfo* output, uint32_t job_index);
uint32_t batch_start(uint32_t batch_end_offset);
uint32_t run_recovery(Model* model, ParameterInfo* output);
