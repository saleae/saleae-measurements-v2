# Saleae Measurements V2

These are a collection of measurement extensions for use with the Saleae Logic software.

Included are basic analog measurements, including vertical measurements like min & max, time measurements like frequency and period, and a few pulse measurements like rise time and overshoot.

This extension package uses Saleae's in-progress second revision measurement API, which has not been released for public use yet. The API used here is expected to change before official release.

For more information about the API, and how to use it, please review:

[Logic 2.3.38 change log](https://discuss.saleae.com/t/logic-2-3-48/1533)

[Measurements API 1.1.0 Documentation](https://saleae.github.io/apidocs/)

_Note, the location of this information is likely to change, and we'll try to update this readme when it does. However, if the links are broken, please check the [discuss site](https://discuss.saleae.com/) for further information, until the 1.1.0 API release becomes official. At that time, the documentation and more information will be available on our [support site](https://support.saleae.com/)._

## Required Software and Environment Variable

This extension uses the second revision measurement API, which is not official yet and only available behind an optional feature flag in the 2.3.48 (and newer) release.

To use, you need to start the logic software from the command prompt:

### Windows (CMD)

```batch
cd "\Program Files\Logic"
set ENABLE_MEASUREMENTS_V2=1
Logic.exe
```

### Windows (PowerShell)

```ps
cd 'C:\Program Files\Logic\'
$Env:ENABLE_MEASUREMENTS_V2=1
.\Logic.exe
```

### MacOS

```bash
cd /Applications/
export ENABLE_MEASUREMENTS_V2=1
./Logic2.app/Contents/MacOS/Logic
```

### Linux

```bash
# cd to directory containing the Logic AppImage
export ENABLE_MEASUREMENTS_V2=1
./Logic-2.3.48-master.AppImage
```

## Voltage Measurements

- Peak to peak
- Maximum
- Minimum
- Mean
- Midpoint
- Amplitude
- Top
- Base
- Root Mean Square (RMS)

## Pulse Measurements

- Rise Time
- Fall Time
- Positive Overshoot
- Negative Overshoot

Rise time and fall time are measured from the 10% crossing point to the 90% crossing point.

## Time Measurements

- Period
- Frequency
- Positive Width
- Negative Width
- Positive Duty Cycle
- Negative Duty Cycle
- Cycle Root Mean Square (RMS)
