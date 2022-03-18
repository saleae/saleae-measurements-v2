# Saleae Measurements V2

These are a collection of measurement extensions for use with the Saleae Logic software.

Included are basic analog measurements, including vertical measurements like min & max, time measurements like frequency and period, and a few pulse measurements like rise time and overshoot.

This extension package uses Saleae's in-progress V2 measurement API, which has not been released for public use yet. The API used here is expected to change before official release.

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
