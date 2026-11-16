#include "DT0_model.h"

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
    float co2_ppm_delta_30)
{
    if (co2_ppm <= 606f)
    {
        if (co2_ppm_delta_10 <= 5f)
        {
            if (co2_ppm_max_30 <= 590f)
            {
                if (voc_index <= 71f)
                {
                    if (co2_ppm_max_30 <= 433f)
                    {
                        return true;
                    }
                    else
                    {
                        return true;
                    }
                }
                else
                {
                    if (pm1_0 <= 5.9f)
                    {
                        return false;
                    }
                    else
                    {
                        return true;
                    }
                }
            }
            else
            {
                if (pm10_0_max_30 <= 19.1f)
                {
                    if (co2_ppm <= 593.5f)
                    {
                        return false;
                    }
                    else
                    {
                        return false;
                    }
                }
                else
                {
                    if (pm1_0 <= 13.4f)
                    {
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }
            }
        }
        else
        {
            if (pm1_0 <= 12.35f)
            {
                if (voc_index_mean_5 <= 8.47411024f)
                {
                    return false;
                }
                else
                {
                    return false;
                }
            }
            else
            {
                return true;
            }
        }
    }
    else
    {
        if (pm10_0_mean_30 <= 17.8350467f)
        {
            if (pm2_5_mean_30 <= 11.6818625f)
            {
                if (pm10_0_mean_10 <= 0.22f)
                {
                    if (co2_ppm_max_30 <= 742.5f)
                    {
                        return false;
                    }
                    else
                    {
                        return false;
                    }
                }
                else
                {
                    return false;
                }
            }
            else
            {
                if (pm10_0_max_30 <= 16f)
                {
                    if (pm1_0 <= 10.05f)
                    {
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }
                else
                {
                    if (pm1_0 <= 16f)
                    {
                        return false;
                    }
                    else
                    {
                        return false;
                    }
                }
            }
        }
        else
        {
            if (voc_index_mean_30 <= 74.373189f)
            {
                if (pm1_0 <= 19.4f)
                {
                    if (pm10_0_mean_30 <= 20.4811192f)
                    {
                        return false;
                    }
                    else
                    {
                        return true;
                    }
                }
                else
                {
                    if (voc_index_delta_10 <= -14.5f)
                    {
                        return false;
                    }
                    else
                    {
                        return false;
                    }
                }
            }
            else
            {
                if (voc_index_mean_30 <= 442.973058f)
                {
                    if (co2_ppm_delta_30 <= 214.902362f)
                    {
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }
                else
                {
                    if (pm2_5_mean_5 <= 13.32f)
                    {
                        return false;
                    }
                    else
                    {
                        return false;
                    }
                }
            }
        }
    }
}
