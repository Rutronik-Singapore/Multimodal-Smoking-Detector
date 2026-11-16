#ifndef DT2_MODEL_H
#define DT2_MODEL_H

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
 * - pm1_0_mean_5s
 * - pm1_0_mean_10s
 * - pm1_0_mean_30s
 * - pm2_5_mean_5s
 * - pm2_5_mean_10s
 * - pm2_5_mean_30s
 * - pm4_0_mean_5s
 * - pm4_0_mean_10s
 * - pm4_0_mean_30s
 * - pm10_0_mean_5s
 * - pm10_0_mean_10s
 * - pm10_0_mean_30s
 * - voc_index_mean_5s
 * - voc_index_mean_10s
 * - voc_index_mean_30s
 * - nox_index_mean_5s
 * - nox_index_mean_10s
 * - nox_index_mean_30s
 * - co2_ppm_mean_5s
 * - co2_ppm_mean_10s
 * - co2_ppm_mean_30s
 *
 * Return: false/0 = no smoking, true/1 = smoking.
 */
bool DT2_model_predict_smoking(float pm1_0,
    float pm2_5,
    float pm4_0,
    float pm10_0,
    float voc_index,
    float nox_index,
    float co2_ppm,
    float pm1_0_mean_5s,
    float pm1_0_mean_10s,
    float pm1_0_mean_30s,
    float pm2_5_mean_5s,
    float pm2_5_mean_10s,
    float pm2_5_mean_30s,
    float pm4_0_mean_5s,
    float pm4_0_mean_10s,
    float pm4_0_mean_30s,
    float pm10_0_mean_5s,
    float pm10_0_mean_10s,
    float pm10_0_mean_30s,
    float voc_index_mean_5s,
    float voc_index_mean_10s,
    float voc_index_mean_30s,
    float nox_index_mean_5s,
    float nox_index_mean_10s,
    float nox_index_mean_30s,
    float co2_ppm_mean_5s,
    float co2_ppm_mean_10s,
    float co2_ppm_mean_30s);

#endif
