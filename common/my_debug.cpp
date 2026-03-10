#include <cstddef>
#include <cstring>
#include <cinttypes> // for PRId64
#include <cstdint>
#include <memory>
#include "config.h"
#include "counters.h"
#include "data.h"
#include "my_debug.h"
#include "cnn_common.h"
#include "data_structures.h"
#include "double_buffering.h"
#include "intermittent-cnn.h"
#include "layers.h"
#include "my_dsplib.h"
#include "op_utils.h"
#include "platform.h"
#ifdef USE_PROTOBUF
#include "model_output.pb.h"
#endif

uint8_t dump_integer = 1;

template<>
void my_printf_wrapper() {}

#ifdef USE_PROTOBUF
std::unique_ptr<ModelOutput> model_output_data;
#endif

#define PRINT_NEWLINE_IF_DATA_NOT_SAVED if (!layer_out) { my_printf(NEWLINE); }

ValueInfo::ValueInfo(const ParameterInfo *cur_param, Model *model) {
    this->scale = cur_param->scale.toFloat();
}

static void print_q15(LayerOutput* layer_out, int16_t val, const ValueInfo& val_info) {
    uint8_t use_prefix = 0;
    float real_value = q15_to_float(val, val_info, &use_prefix);
#ifdef USE_PROTOBUF
    if (layer_out) {
        layer_out->add_value(real_value);
    } else
#endif
    if (dump_integer) {
        my_printf("% 6d ", val);
    } else {
        my_printf(use_prefix ? "   *% 9.6f" : "% 13.6f", real_value);
    }
}

static void print_integer(LayerOutput* layer_out, int16_t val) {
#ifdef USE_PROTOBUF
    if (layer_out) {
        layer_out->add_value(val);
    } else
#endif
    {
        my_printf("% 6d ", val);
    }
}

void dump_matrix(const int16_t *mat, size_t len, const ValueInfo& val_info, const char* layer_name, const char* op_type) {
    LayerOutput* layer_out = nullptr;
#ifdef USE_PROTOBUF
    if (layer_name && model_output_data.get()) {
        layer_out =  model_output_data->add_layer_out();
        layer_out->set_name(layer_name);
        layer_out->set_op_type(op_type);
        layer_out->add_dims(len);
    }
#endif

#ifndef __arm__
    my_printf("Scale: %f" NEWLINE, val_info.scale);
    MY_ASSERT(val_info.scale != 0);
#endif
    for (size_t j = 0; j < len; j++) {
        print_q15(layer_out, mat[j], val_info);
        if (j && (j % 16 == 15)) {
            my_printf(NEWLINE);
        }
    }
    my_printf(NEWLINE);
}

static void dump_params_common(Model* model, const ParameterInfo* cur_param, const char* layer_name, const char* op_type, LayerOutput** p_layer_out) {
    my_printf("Layer: %s" NEWLINE, layer_name);
    my_printf("Slot: %d" NEWLINE, cur_param->slot);
#ifndef __arm__
    my_printf("Scale: %f" NEWLINE, cur_param->scale.toFloat());
    MY_ASSERT(cur_param->scale.toFloat() != 0);
#endif
    my_printf("Params len: %" PRId32 NEWLINE, cur_param->params_len);
    my_printf("Dims: ");
    for (uint8_t j = 0; j < 4; j++) {
        if (cur_param->dims[j]) {
            my_printf("%d, ", cur_param->dims[j]);
        }
    }
    my_printf(NEWLINE);
    my_printf("Channel last: %s" NEWLINE, cur_param->param_flags & CHANNEL_LAST ? "true" : "false");

#ifdef USE_PROTOBUF
    if (layer_name && model_output_data.get()) {
        LayerOutput* layer_out = *p_layer_out = model_output_data->add_layer_out();
        layer_out->set_name(layer_name);
        layer_out->set_op_type(op_type);
        for (uint8_t idx = 0; idx < 4; idx++) {
            layer_out->add_dims(cur_param->dims[idx]);
        }
    }
#endif
}

static void extract_dimensions(const ParameterInfo* cur_param, uint16_t* NUM, uint16_t* H, uint16_t* W, uint16_t* CHANNEL) {
    if (cur_param->dims[3]) {
        // 4-D tensor, NCHW
        *NUM = cur_param->dims[0];
        *CHANNEL = cur_param->dims[1];
        *H = cur_param->dims[2];
        *W = cur_param->dims[3];
    } else if (cur_param->dims[2]) {
        // 3-D tensor, NCW
        *NUM = cur_param->dims[0];
        *CHANNEL = cur_param->dims[1];
        *H = 1;
        *W = cur_param->dims[2];
    } else if (cur_param->dims[1]) {
        // matrix, HW
        *NUM = *CHANNEL = 1;
        *H = cur_param->dims[0];
        *W = cur_param->dims[1];
    } else {
        // vector, W
        *NUM = *CHANNEL = *H = 1;
        *W = cur_param->dims[0];
    }

    // find real num
    uint32_t expected_params_len = sizeof(int16_t);
    for (uint8_t idx = 0; idx < 4; idx++) {
        if (cur_param->dims[idx]) {
            expected_params_len *= cur_param->dims[idx];
        }
    }
    if (expected_params_len != cur_param->params_len) {
        MY_ASSERT(cur_param->dims[0] == 1);
        *NUM = cur_param->params_len / expected_params_len;
    }
}

static bool check_ofm_dumped(const ParameterInfo* cur_param) {
#if MY_DEBUG >= MY_DEBUG_LAYERS
    uint16_t node_idx = cur_param->parameter_info_idx - N_INPUT;
    NodeFlags* node_flags = get_node_flags(node_idx);

    uint8_t odd_run_counter = get_model()->run_counter % 2;
    uint8_t has_ofm_dumped_flag = static_cast<uint8_t>((node_flags->general_flags & OFM_DUMPED) == OFM_DUMPED);
    if ((odd_run_counter + has_ofm_dumped_flag) % 2 == 1) {
        my_printf("check_ofm_dumped: skipped" NEWLINE);
        return false;
    }

    node_flags->general_flags ^= OFM_DUMPED;
    commit_node_flags(node_flags);
#endif

    return true;
}

void dump_params_nhwc(Model *model, const ParameterInfo *cur_param, const char* layer_name, const char* op_type) {
    uint16_t NUM, H, W, CHANNEL;

    if (!check_ofm_dumped(cur_param)) {
        return;
    }

    disable_counters();
    extract_dimensions(cur_param, &NUM, &H, &W, &CHANNEL);
    LayerOutput* layer_out = nullptr;
    dump_params_common(model, cur_param, layer_name, op_type, &layer_out);
    int16_t output_tile_c = cur_param->dims[1];
    for (uint16_t n = 0; n < NUM; n++) {
        my_printf("Matrix %d" NEWLINE, n);
        for (uint16_t tile_c_base = 0; tile_c_base < CHANNEL; tile_c_base += output_tile_c) {
            uint16_t cur_tile_c = MIN_VAL(output_tile_c, CHANNEL - tile_c_base);
            for (uint16_t c = 0; c < cur_tile_c; c++) {
                if (!layer_out) {
                    my_printf("Channel %d" NEWLINE, tile_c_base + c);
                }
                for (uint16_t h = 0; h < H; h++) {
                    for (uint16_t w = 0; w < W; w++) {
                        size_t offset2;
                        if (cur_param->param_flags & TRANSPOSED) {
                            // internal format is NWHC
                            offset2 = n * W * H * CHANNEL + W * H * tile_c_base + w * H * cur_tile_c + h * cur_tile_c + c;
                        } else {
                            // internal format is NHWC
                            offset2 = n * H * W * CHANNEL + H * W * tile_c_base + h * W * cur_tile_c + w * cur_tile_c + c;
                        }

                        int16_t val = get_q15_param(model, cur_param, offset2);
                        if (cur_param->param_flags & INTEGER) {
                            print_integer(layer_out, val);
                        } else {
                            print_q15(layer_out, val, ValueInfo(cur_param, model));
                        }
                    }
                    PRINT_NEWLINE_IF_DATA_NOT_SAVED
                }
                PRINT_NEWLINE_IF_DATA_NOT_SAVED
            }
        }
        PRINT_NEWLINE_IF_DATA_NOT_SAVED
    }
    enable_counters();
}

// dump in NCHW format
void dump_params(Model *model, const ParameterInfo *cur_param, const char* layer_name, const char* op_type) {
    uint16_t NUM, H, W, CHANNEL;

    if (!check_ofm_dumped(cur_param)) {
        return;
    }

    disable_counters();
    extract_dimensions(cur_param, &NUM, &H, &W, &CHANNEL);
    LayerOutput* layer_out = nullptr;
    dump_params_common(model, cur_param, layer_name, op_type, &layer_out);
    for (uint16_t i = 0; i < NUM; i++) {
        my_printf("Matrix %d" NEWLINE, i);
        for (uint16_t j = 0; j < CHANNEL; j++) {
            if (!layer_out) {
                my_printf("Channel %d" NEWLINE, j);
            }
            for (uint16_t k = 0; k < H; k++) {
                for (uint16_t l = 0; l < W; l++) {
                    // internal format is NCHW
                    size_t offset = i * H * W * CHANNEL + j * H * W + k * W + l;
                    int16_t val = get_q15_param(model, cur_param, offset);
                    if (cur_param->param_flags & INTEGER) {
                        print_integer(layer_out, val);
                    } else {
                        print_q15(layer_out, val, ValueInfo(cur_param, model));
                    }
                }
                PRINT_NEWLINE_IF_DATA_NOT_SAVED
            }
            PRINT_NEWLINE_IF_DATA_NOT_SAVED
        }
        PRINT_NEWLINE_IF_DATA_NOT_SAVED
    }
    enable_counters();
}

void dump_turning_points(Model *model, const ParameterInfo *output) {
}

void dump_matrix(const int16_t *mat, size_t rows, size_t cols, const ValueInfo& val_info) {
#ifndef __arm__
    my_printf("Scale: %f", val_info.scale);
    MY_ASSERT(val_info.scale != 0);
#endif
    if (rows > cols) {
        my_printf(" (transposed)" NEWLINE);
        for (size_t j = 0; j < cols; j++) {
            for (size_t i = 0; i < rows; i++) {
                size_t offset = i * cols + j;
                print_q15(nullptr, mat[offset], val_info);
            }
            my_printf(NEWLINE);
        }
    } else {
        my_printf(NEWLINE);
        for (size_t j = 0; j < rows * cols; j++) {
            print_q15(nullptr, mat[j], val_info);
            if ((j+1) % cols == 0) {
                my_printf(NEWLINE);
            }
        }
    }
    my_printf(NEWLINE);
}

#if MY_DEBUG >= MY_DEBUG_NORMAL && DYNAMIC_DNN_APPROACH != DYNAMIC_DNN_FINE_GRAINED

static const uint16_t BUFFER_TEMP_SIZE = 16;
static int16_t buffer_temp[BUFFER_TEMP_SIZE];

void compare_vm_nvm_impl(int16_t* vm_data, Model* model, const ParameterInfo* output, uint32_t output_offset, uint16_t blockSize) {
    disable_counters();
    check_buffer_address(vm_data, blockSize);

    memset(buffer_temp, 0, BUFFER_TEMP_SIZE * sizeof(int16_t));
    for (uint16_t offset = 0; offset < blockSize; offset += BUFFER_TEMP_SIZE) {
        uint16_t cur_block_size = MIN_VAL(BUFFER_TEMP_SIZE, blockSize - offset);
        my_memcpy_from_param(model, buffer_temp, output, output_offset + offset, cur_block_size * sizeof(int16_t));
        for (uint16_t idx = 0; idx < cur_block_size; idx++) {
            MY_ASSERT_ALWAYS(vm_data[idx + offset] == buffer_temp[idx]);
        }
    }
    enable_counters();
}

#else

void compare_vm_nvm_impl(int16_t* vm_data, Model* model, const ParameterInfo* output, uint32_t output_offset, uint16_t blockSize) {}

#endif

void check_nvm_write_address_impl(uint32_t nvm_offset, size_t n) {
    if (nvm_offset >= INTERMEDIATE_PARAMETERS_INFO_OFFSET && nvm_offset < MODEL_OFFSET) {
        MY_ASSERT((nvm_offset - INTERMEDIATE_PARAMETERS_INFO_OFFSET) % sizeof(ParameterInfo) == 0);
    } else if (nvm_offset < INTERMEDIATE_PARAMETERS_INFO_OFFSET) {
        MY_ASSERT(n <= INTERMEDIATE_PARAMETERS_INFO_OFFSET - nvm_offset, "Size %d too large!!! nvm_offset=%d" NEWLINE, n, nvm_offset);
    }
}

void dump_footprints(uint16_t layer_idx) {
#if HAWAII && (DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS_BASIC || DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS) && MY_DEBUG >= MY_DEBUG_VERBOSE
    for (uint8_t copy_id = 0; copy_id < 2; copy_id++) {
        Footprint tmp_footprint;

        read_from_nvm(&tmp_footprint, nvm_addr<Footprint>(copy_id, layer_idx), sizeof(Footprint));

        if (copy_id == 0) {
            uint8_t cached_copy_id = *copy_id_cache_addr<Footprint>();
            my_printf_debug("Footprint NVM pointer=%d, cached pointer=%d" NEWLINE, tmp_footprint.version, cached_copy_id);
            MY_ASSERT(cached_copy_id == 0 || cached_copy_id == tmp_footprint.version + 1);
        }

        my_printf_debug("Footprint copy %d NVM values=", copy_id);
        for (const uint8_t& value : tmp_footprint.values) {
            my_printf_debug("%d ", value);
        }
#if DYNAMIC_DNN_APPROACH == DYNAMIC_DNN_MULTIPLE_INDICATORS
        const UnshuffledFootprint& mirror = unshuffled_footprint_mirror[copy_id];
        my_printf_debug(" mirror values=%d %d %d",
                        mirror.values[FootprintOffset::NUM_COMPLETED_JOBS],
                        mirror.values[FootprintOffset::COMPUTATION_UNIT_INDEX],
                        mirror.values[FootprintOffset::NUM_SKIPPED_JOBS]);
#endif
        my_printf_debug(NEWLINE);
    }
    my_printf_debug("Footprint VM values=");
    for (const uint8_t& value : vm_addr<Footprint>(layer_idx)->values) {
        my_printf_debug("%d ", value);
    }
    my_printf_debug(NEWLINE);
#endif
}

