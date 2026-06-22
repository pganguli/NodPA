// Stdin reader for parse_model.c.
//
// read_buffer: reads until EOF or until max_length bytes have been consumed,
//   returning the number of bytes read.  Returns 0 if the limit is exceeded
//   (with a message to stderr) so the caller can abort cleanly.
//
// MAX_MSG_SIZE: hard limit on ONNX binary size (128 MB).  Adjust if you
//   need to inspect very large models.

#pragma once

size_t read_buffer(unsigned max_length, uint8_t* out);

#define MAX_MSG_SIZE (128 * 1024 * 1024)
