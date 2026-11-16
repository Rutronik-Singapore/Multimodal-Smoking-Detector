#ifndef DT1_MODEL_H
#define DT1_MODEL_H

#include <stdbool.h>

/*
 * Feature order:
 * - pm1_0
 * - pm2_5
 * - pm4_0
 * - pm10_0
 * - voc_index
 * - nox_index
 * - co2_ppm
 *
 * Return: false/0 = no smoking, true/1 = smoking.
 */
bool DT1_model_predict_smoking(float pm1_0,
    float pm2_5,
    float pm4_0,
    float pm10_0,
    float voc_index,
    float nox_index,
    float co2_ppm);

#endif
