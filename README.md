# Multimodal Smoking Detector

A research prototype on the **Infineon KIT_PSE84_AI** that investigates two ways of detecting smoking-related activity outdoors: environmental sensing with a Sensirion SEN66 and on-device vision with a USB camera. The repository brings together the embedded firmware, the Bluetooth monitoring app, and the environmental machine-learning evaluation.

## Repository guide

| Folder | What it contains | Where to start |
| --- | --- | --- |
| [`Firmware/`](Firmware/) | ModusToolbox workspace for the PSoC Edge E84 board, including SEN66 acquisition, XENSIV radar, the deployed decision tree, camera inference, and BLE. | [Firmware setup and programming](Firmware/README.md) |
| [`User Interface/`](User%20Interface/) | .NET MAUI app for live readings, charts, manual smoking labels, and CSV recording over BLE. | [App setup and recording](User%20Interface/README.md) |
| [`Evaluation/`](Evaluation/) | Recorded SEN66 data, preparation and training scripts, model exports, and the saved comparison of three Decision Tree variants and two DEEPCRAFT models. | [Evaluation workflow](Evaluation/README.md) · [Five-model report](Evaluation/academic_study/DEEPCRAFT_training/reports/five_model_report.md) |

## How the system fits together

- **Environmental sensing on CM33.** The non-secure application reads the SEN66, runs the deployed environmental decision tree, and sends measurements and predictions over BLE. XENSIV radar adds presence and distance information to the system. Radar distance is not an input to the deployed decision tree.
- **Camera inference on CM55.** The camera path runs separately and draws detections on the kit display. The BLE dashboard does not receive camera frames or display the vision results. Its **Predicted** field shows the environmental model's result, while **Manual Label** records the operator's annotation.
- **Offline evaluation.** The Python scripts compare environmental models using the bundled recordings and frozen session partitions. The five-model comparison does not include the camera model or measure the accuracy of a fused system.

## Get started on Windows

Clone this organization repository once, or open your existing checkout in GitHub Desktop. The component READMEs include standalone clone examples for the original repositories. If you are working from this checkout, use the folders below instead of cloning them again.

```powershell
git clone -c core.longpaths=true https://github.com/Rutronik-Singapore/Multimodal-Smoking-Detector.git
Set-Location .\Multimodal-Smoking-Detector
```

1. **Run the hardware demo.** Open ModusToolbox's **modus-shell** in `Firmware/` and follow the build and programming guide for the KIT_PSE84_AI. Connect the SEN66 for environmental measurements and a compatible USB camera and display for the vision demo. [Firmware guide](Firmware/README.md)
2. **Monitor and record.** Open [`BLE_APP.slnx`](User%20Interface/BLE_APP.slnx) in Visual Studio with the .NET 10 MAUI tooling. Run the app on Windows or a compatible Android device, scan for **PSE84-IAQ**, and connect. Select a logging folder if you want to save a CSV. [App guide](User%20Interface/README.md)
3. **Inspect or reproduce the study.** Start with the saved report, or open PowerShell in `Evaluation/` and follow the Python instructions. Reproducing the saved comparison does not require the board, DEEPCRAFT Studio, or retraining. If you merge new logs, explicitly check which CSV the next script reads. A new date-prefixed merge does not replace the bundled study inputs. [Evaluation guide](Evaluation/README.md)

## Scope

The evaluation reflects the recorded study sessions, which were collected under uncontrolled conditions. It is useful for comparing environmental modelling approaches on that dataset, not for claiming field accuracy. The prototype is not a certified smoke or fire alarm.
