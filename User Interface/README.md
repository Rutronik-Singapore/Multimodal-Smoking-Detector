# PSE84 Air Quality Monitor

A .NET MAUI companion app for the [PSoC Edge E84 Air Quality firmware](https://github.com/Jifsopeon/kit_pse84_ai_iaq/tree/master). It connects to the board over Bluetooth Low Energy (BLE), plots live environmental readings, and logs sensor samples and smoking labels for later analysis.

## What the dashboard shows

- SEN66 readings for particulate matter, humidity, temperature, VOC index, NOx index, and CO₂, plus radar distance from the firmware.
- Separate fields for the **Manual Label** and **Predicted** result. The manual label is the operator's annotation. The prediction comes from the board's environmental model. The app decodes and displays it without running inference itself.
- A **Logging Folder** selector and logging status. The app writes live samples locally to CSV. It does not receive camera frames or show the board's vision bounding boxes.

The current app opens directly to the air quality dashboard. Older project/task-management files remain in the repository but are excluded from the active app build.

## Build and run on Windows

Install the **.NET 10 SDK** and a Visual Studio installation with .NET MAUI support. For Android deployment, install the Android SDK and JDK through the MAUI tooling. Use a physical BLE-capable Android device or a Windows PC with a working BLE adapter for end-to-end checks.

```powershell
git clone --branch master https://github.com/Jifsopeon/BLE_APP.git
Set-Location BLE_APP
```

Open [`BLE_APP.slnx`](BLE_APP.slnx) in Visual Studio, select **BLE_APP** as the startup project, and choose **Windows Machine** or a connected **Android device**. For a USB-connected Android device, enable USB debugging and authorize the PC on the device before deploying.

The [project file](BLE_APP.csproj) declares Android API 33 (**Android 13**) as the minimum for the Android build and uses MSIX packaging for Windows. It also declares iOS and Mac Catalyst targets, but the repository's public-folder CSV storage is implemented for **Android and Windows**. The Apple targets should not be assumed to have the same logging support.

## Connect and record

1. Power a board running the companion firmware. Turn on Bluetooth and grant the app's Bluetooth or **Nearby devices** permissions when prompted.
2. In the app, use **Select Folder** under **Logging Folder** and pick a writable location. On Android, use the system folder picker. On Windows, check the selected folder again after restarting the app.
3. Select **Scan**, choose **PSE84-IAQ**, and select **Connect**. The app subscribes to the firmware stream. A valid sample can start a CSV log automatically when a folder is selected.
4. Use **No Smoking** or **Smoking** to request a manual-label change. Wait for the streamed state to confirm the change before relying on it as an annotation.
5. Select **Disconnect** at the end of the recording to close the session log. Logs use numbered filenames such as `Log0001.csv` in the selected folder. Each row includes a sample number, time and elapsed time, sensor values, distance, the manual label, and the firmware prediction.

If the board does not appear, check that it is advertising and that scanning permissions are granted. If it connects but no readings arrive, make sure the firmware and app are from matching versions of the project. If readings arrive but no CSV is written, check **Logging Folder** and its status. Android includes a connected-device foreground service for ongoing BLE work. Check notification permission and device background restrictions if streaming stops when the app leaves the foreground.

## Source map and tests

| Path | Role |
| --- | --- |
| [`MauiProgram.cs`](MauiProgram.cs), [`Pages/MainPage.xaml`](Pages/MainPage.xaml), [`PageModels/MainPageModel.cs`](PageModels/MainPageModel.cs) | App composition, dashboard, and UI commands. |
| [`Services/BluetoothSensorService.cs`](Services/BluetoothSensorService.cs) | BLE scan, connect, discovery, notifications, and manual-label commands. |
| [`Services/SensorPacketProtocol.cs`](Services/SensorPacketProtocol.cs) and [`Services/SensorPacketDecoder.cs`](Services/SensorPacketDecoder.cs) | Shared data interpretation between the app and firmware. |
| [`Services/SensorLogService.cs`](Services/SensorLogService.cs) and [`Services/SensorCsvFormatter.cs`](Services/SensorCsvFormatter.cs) | CSV session lifecycle and output formatting. |
| [`Platforms/Android/`](Platforms/Android/) | Android permissions and BLE foreground service. |
| [`ThirdParty/Plugin.Bluetooth/`](ThirdParty/Plugin.Bluetooth/) | Vendored Bluetooth adapters. See the [component license](ThirdParty/Plugin.Bluetooth/LICENSE.md). |

Run the executable test project from the repository root. It exercises the decoder and selected CSV, log-naming, and view-model behavior without connecting to hardware.

```powershell
dotnet run --project .\Tests\BLE_APP.DecoderTests\BLE_APP.DecoderTests.csproj
```

Use the separate **Definitive Evaluation** archive for SEN66 data preparation and model comparison. This app is the collection and monitoring interface.
