#ifndef DT2_FEATURES_H
#define DT2_FEATURES_H

#include <stdbool.h>
#include <stdint.h>

#define DT2_FEATURE_COUNT (28U)
#define DT2_HISTORY_SIZE (31U)

void DT2_features_reset(void);
bool DT2_features_update(float pm1_0, float pm2_5, float pm4_0, float pm10_0,
                         float voc_index, float nox_index, float co2_ppm,
                         float features[DT2_FEATURE_COUNT]);

#endif
