#include "DT2_model.h"

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
    float co2_ppm_mean_30s)
{
    if (co2_ppm <= 612f)
    {
        if (voc_index <= 71f)
        {
            if (co2_ppm_mean_30s <= 556.716493f)
            {
                if (pm2_5_mean_30s <= 9.68333333f)
                {
                    if (nox_index <= 1.5f)
                    {
                        if (voc_index_mean_30s <= 20.5143913f)
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
                        if (pm10_0_mean_30s <= 13.6466667f)
                        {
                            return true;
                        }
                        else
                        {
                            return false;
                        }
                    }
                }
                else
                {
                    if (pm1_0_mean_5s <= 4.896f)
                    {
                        if (pm1_0 <= 4.75f)
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
                        if (nox_index_mean_10s <= 224.517043f)
                        {
                            return true;
                        }
                        else
                        {
                            return true;
                        }
                    }
                }
            }
            else
            {
                if (pm1_0 <= 13.3f)
                {
                    if (voc_index <= 5f)
                    {
                        return true;
                    }
                    else
                    {
                        if (pm10_0 <= 18.7f)
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
                    if (pm10_0_mean_30s <= 26.1363739f)
                    {
                        return false;
                    }
                    else
                    {
                        if (pm1_0_mean_10s <= 18.4895826f)
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
        }
        else
        {
            if (pm1_0 <= 6.1f)
            {
                if (nox_index <= 1f)
                {
                    return false;
                }
                else
                {
                    if (nox_index <= 6f)
                    {
                        if (co2_ppm_mean_30s <= 451.766667f)
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
                        return false;
                    }
                }
            }
            else
            {
                if (nox_index <= 6.5f)
                {
                    if (nox_index_mean_30s <= 5f)
                    {
                        if (nox_index <= 3.5f)
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
                        if (co2_ppm_mean_30s <= 505.799638f)
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
                    if (pm1_0 <= 9.60947826f)
                    {
                        return true;
                    }
                    else
                    {
                        return true;
                    }
                }
            }
        }
    }
    else
    {
        if (pm10_0_mean_30s <= 18.3558783f)
        {
            if (pm10_0_mean_30s <= 17.4885333f)
            {
                return false;
            }
            else
            {
                if (co2_ppm <= 640.921739f)
                {
                    if (pm1_0 <= 7.15f)
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
                    if (co2_ppm_mean_5s <= 708.252261f)
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
            if (nox_index_mean_30s <= 1.66666667f)
            {
                if (pm1_0 <= 6.6f)
                {
                    if (co2_ppm <= 742.575217f)
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
                    if (pm10_0_mean_30s <= 34.1684754f)
                    {
                        if (voc_index_mean_30s <= 1f)
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
                        if (pm4_0_mean_5s <= 29.317f)
                        {
                            return false;
                        }
                        else
                        {
                            return true;
                        }
                    }
                }
            }
            else
            {
                if (nox_index <= 6f)
                {
                    if (co2_ppm_mean_30s <= 637.350464f)
                    {
                        return true;
                    }
                    else
                    {
                        if (co2_ppm <= 782.29913f)
                        {
                            return true;
                        }
                        else
                        {
                            return true;
                        }
                    }
                }
                else
                {
                    return false;
                }
            }
        }
    }
}
