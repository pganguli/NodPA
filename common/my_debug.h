// Debug output, assertion, and tensor-dump utilities.
//
// MY_ASSERT(cond[, fmt, ...])
//   Stripped to a no-op when MY_DEBUG < MY_DEBUG_NORMAL (i.e., production
//   MCU builds).  Compile in NO_ASSERT mode to recover the tiny code-space.
//   Use MY_ASSERT_ALWAYS for checks that must survive even in release builds.
//
// Verbosity macros — each symbol expands to the real function when the debug
// level is high enough, or to a no-op variadic macro otherwise:
//   MY_DEBUG_VERBOSE  → dump_matrix_debug, my_printf_debug,
//                        dump_turning_points_debug, dump_footprints_debug
//   MY_DEBUG_LAYERS   → dump_params_debug, dump_params_nhwc_debug
//   MY_DEBUG_NORMAL   → compare_vm_nvm, check_nvm_write_address
//
// ValueInfo carries the Q15 scale factor needed to pretty-print raw int16_t
// values as human-readable floats in dump_matrix / dump_params.
//
// my_printf is backed by print2uart_new on MCUs and printf on the PC
// simulator.  NEWLINE is "\r\n" on MCU and "\n" on PC to match UART
// terminal expectations.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#include "config.h"
#include "data.h"
#include "platform.h"

#if defined(__MSP430__) || defined(__MSP432__)
#include "tools/myuart.h"
#define my_printf print2uart_new
#define my_flush()
#define NEWLINE "\r\n"
#else
#include <cstdio>
#define my_printf printf
#define my_flush()  \
  do {              \
    fflush(stdout); \
  } while (0);
#define NEWLINE "\n"
#endif

template <typename... Args>
void my_printf_wrapper(Args... args) {
  my_printf(args...);
}

template <>
void my_printf_wrapper();

template <typename... Args>
void my_assert_impl(const char* file, uint16_t line, uint8_t cond,
                    Args... args) {
  if (!cond) {
    uint16_t path_len = strlen(file);
    int16_t idx = path_len - 1;
    while (idx > 0) {
      if (file[idx] == '/') {
        break;
      }
      idx--;
    }

    my_printf("Assertion failed at %s:%d" NEWLINE, file + (idx + 1), line);
    my_printf_wrapper(args...);
    ERROR_OCCURRED();
  }
}

#if MY_DEBUG >= MY_DEBUG_NORMAL
#define MY_ASSERT(...) my_assert_impl(__FILE__, __LINE__, __VA_ARGS__)
#else
#define MY_ASSERT(...)
#endif
// for checks that need to run when MY_DEBUG == 0
#define MY_ASSERT_ALWAYS(...) my_assert_impl(__FILE__, __LINE__, __VA_ARGS__)

struct ParameterInfo;
struct Model;
struct Node;

struct ValueInfo {
  ValueInfo(const ParameterInfo* cur_param, Model* model = nullptr);
  ValueInfo() = delete;
  explicit ValueInfo(float _scale) : scale(_scale) {}

  float scale;
};

extern uint8_t dump_integer;

class ModelOutput;
class LayerOutput;
extern std::unique_ptr<ModelOutput> model_output_data;

void dump_matrix(const int16_t* mat, size_t len, const ValueInfo& val_info,
                 const char* layer_name = nullptr,
                 const char* op_type = nullptr);
void dump_matrix(const int16_t* mat, size_t rows, size_t cols,
                 const ValueInfo& val_info);
void dump_params(Model* model, const ParameterInfo* cur_param,
                 const char* layer_name, const char* op_type);
void dump_params_nhwc(Model* model, const ParameterInfo* cur_param,
                      const char* layer_name, const char* op_type);
void dump_turning_points(Model* model, const ParameterInfo* output);
void compare_vm_nvm_impl(int16_t* vm_data, Model* model,
                         const ParameterInfo* output, uint32_t output_offset,
                         uint16_t blockSize);
void check_nvm_write_address_impl(uint32_t nvm_offset, size_t n);
void dump_footprints(uint16_t layer_idx);

#if MY_DEBUG >= MY_DEBUG_VERBOSE

#define dump_matrix_debug dump_matrix
#define my_printf_debug my_printf
#define dump_turning_points_debug dump_turning_points
#define dump_footprints_debug dump_footprints

#else

#define dump_matrix_debug(...)
#define dump_model_debug(...)
#define my_printf_debug(...)
#define dump_turning_points_debug(...)
#define dump_footprints_debug(...)

#endif

#if MY_DEBUG >= MY_DEBUG_LAYERS

#define dump_params_debug dump_params
#define dump_params_nhwc_debug dump_params_nhwc

#else

#define dump_params_debug(...)
#define dump_params_nhwc_debug(...)

#endif

#if MY_DEBUG >= MY_DEBUG_NORMAL

#define compare_vm_nvm compare_vm_nvm_impl
#define check_nvm_write_address check_nvm_write_address_impl

#else

#define compare_vm_nvm(...)
#define check_nvm_write_address(...)

#endif
