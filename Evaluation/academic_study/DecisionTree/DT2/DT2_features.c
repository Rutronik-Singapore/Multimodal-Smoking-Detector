#include "DT2_features.h"

#include <stddef.h>

typedef struct
{
    float pm1_0[DT2_HISTORY_SIZE];
    float pm2_5[DT2_HISTORY_SIZE];
    float pm4_0[DT2_HISTORY_SIZE];
    float pm10_0[DT2_HISTORY_SIZE];
    float voc_index[DT2_HISTORY_SIZE];
    float nox_index[DT2_HISTORY_SIZE];
    float co2_ppm[DT2_HISTORY_SIZE];
    uint32_t total_count;
    uint8_t next_index;
} dt2_history_t;

static dt2_history_t state;

void DT2_features_reset(void)
{
    state.total_count = 0U;
    state.next_index = 0U;
}

static float get_age(const float *buf, uint8_t age)
{
    uint8_t idx = (uint8_t)((state.next_index + DT2_HISTORY_SIZE - 1U - age) % DT2_HISTORY_SIZE);
    return buf[idx];
}

static float mean_recent(const float *buf, uint8_t window)
{
    float sum = 0.0f;
    for (uint8_t i = 0U; i < window; i++) { sum += get_age(buf, i); }
    return sum / (float)window;
}

static float max_recent(const float *buf, uint8_t window)
{
    float value = get_age(buf, 0U);
    for (uint8_t i = 1U; i < window; i++) { float v = get_age(buf, i); if (v > value) { value = v; } }
    return value;
}

static float delta_recent(const float *buf, float current_value, uint8_t window)
{
    return current_value - get_age(buf, window);
}

bool DT2_features_update(float pm1_0, float pm2_5, float pm4_0, float pm10_0,
                         float voc_index, float nox_index, float co2_ppm,
                         float features[DT2_FEATURE_COUNT])
{
    state.pm1_0[state.next_index] = pm1_0;
    state.pm2_5[state.next_index] = pm2_5;
    state.pm4_0[state.next_index] = pm4_0;
    state.pm10_0[state.next_index] = pm10_0;
    state.voc_index[state.next_index] = voc_index;
    state.nox_index[state.next_index] = nox_index;
    state.co2_ppm[state.next_index] = co2_ppm;
    state.next_index = (uint8_t)((state.next_index + 1U) % DT2_HISTORY_SIZE);
    state.total_count++;
    if (state.total_count <= 30U) { return false; }
    uint16_t i = 0U;
    features[i++] = pm1_0;
    features[i++] = pm2_5;
    features[i++] = pm4_0;
    features[i++] = pm10_0;
    features[i++] = voc_index;
    features[i++] = nox_index;
    features[i++] = co2_ppm;
    features[i++] = mean_recent(state.pm1_0, 5U);
    features[i++] = mean_recent(state.pm1_0, 10U);
    features[i++] = mean_recent(state.pm1_0, 30U);
    features[i++] = mean_recent(state.pm2_5, 5U);
    features[i++] = mean_recent(state.pm2_5, 10U);
    features[i++] = mean_recent(state.pm2_5, 30U);
    features[i++] = mean_recent(state.pm4_0, 5U);
    features[i++] = mean_recent(state.pm4_0, 10U);
    features[i++] = mean_recent(state.pm4_0, 30U);
    features[i++] = mean_recent(state.pm10_0, 5U);
    features[i++] = mean_recent(state.pm10_0, 10U);
    features[i++] = mean_recent(state.pm10_0, 30U);
    features[i++] = mean_recent(state.voc_index, 5U);
    features[i++] = mean_recent(state.voc_index, 10U);
    features[i++] = mean_recent(state.voc_index, 30U);
    features[i++] = mean_recent(state.nox_index, 5U);
    features[i++] = mean_recent(state.nox_index, 10U);
    features[i++] = mean_recent(state.nox_index, 30U);
    features[i++] = mean_recent(state.co2_ppm, 5U);
    features[i++] = mean_recent(state.co2_ppm, 10U);
    features[i++] = mean_recent(state.co2_ppm, 30U);
    return true;
}
