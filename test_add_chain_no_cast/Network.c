#include "DeeployBasicMath.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Network.h"

float32_t *DeeployNetwork_input_0;
float32_t *DeeployNetwork_input_1;
float32_t *DeeployNetwork_input_2;
float32_t *DeeployNetwork_output_0;
void *DeeployNetwork_inputs[3];
void *DeeployNetwork_outputs[1];
void RunNetwork(__attribute__((unused)) uint32_t core_id,
                __attribute__((unused)) uint32_t numThreads) {
  float32_t *DeeployNetwork_X_tensor;
  DeeployNetwork_X_tensor = (float32_t *)deeploy_malloc(4 * 125);

  // Add (Name: Add1, Op: Add)
  BEGIN_SINGLE_CORE
  for (uint32_t i = 0; i < 125; i++) {
    DeeployNetwork_X_tensor[i] =
        DeeployNetwork_input_0[i] + DeeployNetwork_input_1[i];
  }
  END_SINGLE_CORE

  // Add (Name: Add2, Op: Add)
  BEGIN_SINGLE_CORE
  for (uint32_t i = 0; i < 125; i++) {
    DeeployNetwork_output_0[i] =
        DeeployNetwork_X_tensor[i] + DeeployNetwork_input_2[i];
  }
  END_SINGLE_CORE

  SINGLE_CORE deeploy_free(DeeployNetwork_X_tensor);
}

void InitNetwork(__attribute__((unused)) uint32_t core_id,
                 __attribute__((unused)) uint32_t numThreads) {
  DeeployNetwork_input_0 = (float32_t *)deeploy_malloc(4 * 125);
  DeeployNetwork_input_1 = (float32_t *)deeploy_malloc(4 * 125);
  DeeployNetwork_input_2 = (float32_t *)deeploy_malloc(4 * 125);
  DeeployNetwork_output_0 = (float32_t *)deeploy_malloc(4 * 125);
  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_inputs[2] = (void *)DeeployNetwork_input_2;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
