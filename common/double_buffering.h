#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include "counters.h"
#include "data.h"
#include "data_structures.h"
#include "my_debug.h"
#include "platform.h"

// Templates to be filled by users
template<typename T>
uint32_t nvm_addr(uint8_t, uint16_t);

template<typename T>
T* vm_addr(uint16_t data_idx);

// typeinfo does not always give names I want
template<typename T>
const char* datatype_name(void);

template<typename T>
uint8_t* copy_id_cache_addr(void) {
    // No copy_id cache by default
    return nullptr;
}

#if HAWAII
// Specialization defined in platform.cpp
// Declaration prevents implicit instantiation in other translation units,
// which causes conflicting symbols in MSVC
template<>
uint8_t* copy_id_cache_addr<Footprint>(void);
#endif

template<typename T>
static uint8_t get_newer_copy_id(uint16_t data_idx) {
    uint8_t* copy_id_cache = copy_id_cache_addr<T>();
    // Use 1 for copy 0 and 2 for copy 1. 0, which comes after power resumption, is considerd invalid (not cached yet)
    if (copy_id_cache && *copy_id_cache > 0) {
        return *copy_id_cache - 1;
    }

    uint8_t copy_id;

    add_counter(offsetof(Counters, nvm_read_shadow_data), 2);

    read_from_nvm(&copy_id, nvm_addr<T>(0, data_idx) + offsetof(T, version), sizeof(uint8_t));
    my_printf_debug("Pointer of shadow %s copies for data item %d: %d" NEWLINE, datatype_name<T>(), data_idx, copy_id);

    if (copy_id_cache) {
        *copy_id_cache = copy_id + 1;
    }

    return copy_id;
}

template<typename T>
T* get_versioned_data(uint16_t data_idx) {
    T *dst = vm_addr<T>(data_idx);

    uint8_t newer_copy_id = get_newer_copy_id<T>(data_idx);
#if ENABLE_COUNTERS
    add_counter(offsetof(Counters, nvm_read_shadow_data), sizeof(T));
    my_printf_debug("Recorded %lu bytes of shadow data read from NVM" NEWLINE, sizeof(T));
#endif
    read_from_nvm(dst, nvm_addr<T>(newer_copy_id, data_idx), sizeof(T) - sizeof(uint8_t));
    my_printf_debug("Using %s copy %d" NEWLINE, datatype_name<T>(), newer_copy_id);
    return dst;
}

template<typename T>
void commit_versioned_data(uint16_t data_idx, uint8_t commit_offset = 0, uint16_t num_bytes = 0) {
    uint8_t newer_copy_id = get_newer_copy_id<T>(data_idx);
    uint8_t older_copy_id = newer_copy_id ^ 1;

    T* vm_ptr = vm_addr<T>(data_idx);

    if (!num_bytes) {
        num_bytes = sizeof(T) - sizeof(uint8_t);
    }

#if ENABLE_COUNTERS
    add_counter(offsetof(Counters, nvm_write_shadow_data), num_bytes);
    my_printf_debug("Recorded %lu bytes of shadow data written to NVM" NEWLINE, num_bytes);
#endif
    write_to_nvm(reinterpret_cast<uint8_t*>(vm_ptr) + commit_offset, nvm_addr<T>(older_copy_id, data_idx) + commit_offset, num_bytes);
    my_printf_debug("Committing to %s copy %d" NEWLINE, datatype_name<T>(), older_copy_id);

    write_to_nvm(&older_copy_id, nvm_addr<T>(0, data_idx) + offsetof(T, version), sizeof(uint8_t));
    my_printf_debug("Updating %s pointer = %d" NEWLINE, datatype_name<T>(), older_copy_id);

    uint8_t* copy_id_cache = copy_id_cache_addr<T>();
    if (copy_id_cache) {
        *copy_id_cache = older_copy_id + 1;
    }
}
