#ifndef DT0_MODEL_H
#define DT0_MODEL_H

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
 * - pm2_5_mean_5
 * - pm2_5_max_5
 * - pm2_5_delta_5
 * - pm2_5_mean_10
 * - pm2_5_max_10
 * - pm2_5_delta_10
 * - pm2_5_mean_30
 * - pm2_5_max_30
 * - pm2_5_delta_30
 * - pm10_0_mean_5
 * - pm10_0_max_5
 * - pm10_0_delta_5
 * - pm10_0_mean_10
 * - pm10_0_max_10
 * - pm10_0_delta_10
 * - pm10_0_mean_30
 * - pm10_0_max_30
 * - pm10_0_delta_30
 * - voc_index_mean_5
 * - voc_index_max_5
 * - voc_index_delta_5
 * - voc_index_mean_10
 * - voc_index_max_10
 * - voc_index_delta_10
 * - voc_index_mean_30
 * - voc_index_max_30
 * - voc_index_delta_30
 * - co2_ppm_mean_5
 * - co2_ppm_max_5
 * - co2_ppm_delta_5
 * - co2_ppm_mean_10
 * - co2_ppm_max_10
 * - co2_ppm_delta_10
 * - co2_ppm_mean_30
 * - co2_ppm_max_30
 * - co2_ppm_delta_30
 *
 * Return: false/0 = no smoking, true/1 = smoking.
 */
bool DT0_model_predict_smoking(float pm1_0,
    float pm2_5,
    float pm4_0,
    float pm10_0,
    float voc_index,
    float nox_index,
    float co2_ppm,
    float pm2_5_mean_5,
    float pm2_5_max_5,
    float pm2_5_delta_5,
    float pm2_5_mean_10,
    float pm2_5_max_10,
    float pm2_5_delta_10,
    float pm2_5_mean_30,
    float pm2_5_max_30,
    float pm2_5_delta_30,
    float pm10_0_mean_5,
    float pm10_0_max_5,
    float pm10_0_delta_5,
    float pm10_0_mean_10,
    float pm10_0_max_10,
    float pm10_0_delta_10,
    float pm10_0_mean_30,
    float pm10_0_max_30,
    float pm10_0_delta_30,
    float voc_index_mean_5,
    float voc_index_max_5,
    float voc_index_delta_5,
    float voc_index_mean_10,
    float voc_index_max_10,
    float voc_index_delta_10,
    float voc_index_mean_30,
    float voc_index_max_30,
    float voc_index_delta_30,
    float co2_ppm_mean_5,
    float co2_ppm_max_5,
    float co2_ppm_delta_5,
    float co2_ppm_mean_10,
    float co2_ppm_max_10,
    float co2_ppm_delta_10,
    float co2_ppm_mean_30,
    float co2_ppm_max_30,
    float co2_ppm_delta_30);

#endif
