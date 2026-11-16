#include "DT1_model.h"

bool DT1_model_predict_smoking(float pm1_0,
    float pm2_5,
    float pm4_0,
    float pm10_0,
    float voc_index,
    float nox_index,
    float co2_ppm)
{
    if (co2_ppm <= 610f)
    {
        if (voc_index <= 72f)
        {
            if (co2_ppm <= 554f)
            {
                if (pm2_5 <= 9.7f)
                {
                    if (pm10_0 <= 14.2f)
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
                    if (co2_ppm <= 513f)
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
                if (pm1_0 <= 13.5f)
                {
                    if (pm4_0 <= 13.8183226f)
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
                    if (pm10_0 <= 28.6f)
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
            if (pm1_0 <= 6f)
            {
                if (pm10_0 <= 6.36196774f)
                {
                    if (voc_index <= 350.562903f)
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
                    if (pm1_0 <= 5.9f)
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
                if (nox_index <= 6.5f)
                {
                    if (pm4_0 <= 15.9f)
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
                    if (pm1_0 <= 8.8f)
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
    }
    else
    {
        if (pm10_0 <= 16.7050645f)
        {
            if (pm2_5 <= 10.1f)
            {
                if (pm1_0 <= 0.2f)
                {
                    if (pm1_0 <= 0.05f)
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
                if (nox_index <= 1.5f)
                {
                    return false;
                }
                else
                {
                    if (co2_ppm <= 754.051613f)
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
            if (nox_index <= 1f)
            {
                if (pm1_0 <= 7.1f)
                {
                    if (voc_index <= 0f)
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
                    if (pm10_0 <= 35.3f)
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
                if (nox_index <= 6f)
                {
                    if (co2_ppm <= 1114.5571f)
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
                    return false;
                }
            }
        }
    }
}
