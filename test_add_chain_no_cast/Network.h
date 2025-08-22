
#ifndef __DEEPLOY_HEADER_
#define __DEEPLOY_HEADER_
#include "DeeployBasicMath.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
void RunNetwork(uint32_t core_id, uint32_t numThreads);
void InitNetwork(uint32_t core_id, uint32_t numThread);

extern float32_t *DeeployNetwork_input_0;
static const uint32_t DeeployNetwork_input_0_len = 125;
extern float32_t *DeeployNetwork_input_1;
static const uint32_t DeeployNetwork_input_1_len = 125;
extern float32_t *DeeployNetwork_input_2;
static const uint32_t DeeployNetwork_input_2_len = 125;
extern float32_t *DeeployNetwork_output_0;
static const uint32_t DeeployNetwork_output_0_len = 125;
static const uint32_t DeeployNetwork_num_inputs = 3;
static const uint32_t DeeployNetwork_num_outputs = 1;
extern void *DeeployNetwork_inputs[3];
extern void *DeeployNetwork_outputs[1];
static const uint32_t DeeployNetwork_inputs_bytes[3] = {500, 500, 500};
static const uint32_t DeeployNetwork_outputs_bytes[1] = {500};
#endif
