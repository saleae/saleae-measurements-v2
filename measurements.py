from saleae.measurements import AnalogMeasurer, Measure
from saleae.data import AnalogSpan
from statistics import mode
import math


def rms(data: AnalogSpan):
    # TODO: numpy accelerate
    sum_of_squares = sum(x**2 for x in data)
    rms = math.sqrt(sum_of_squares / len(data))
    return rms

def histogram_mode(data: AnalogSpan, filter):
    hist, bins = data.histogram()
    _count, index = max((count, index) for index, count in enumerate(hist) if filter((bins[index] + bins[index]) / 2))
    return (bins[index] + bins[index + 1]) / 2

# High & Low can be computed several different ways. If this is False, the "mode" method will be used
use_histogram_for_mode = True

def compute_high(data: AnalogSpan, mid: float):
    if use_histogram_for_mode:
        return histogram_mode(data, lambda x: x >= mid)
    return mode(x for x in data if x >= mid)

def compute_low(data: AnalogSpan, mid: float):
    if use_histogram_for_mode:
        return histogram_mode(data, lambda x: x < mid)
    return mode(x for x in data if x < mid)


class VoltageMeasurer(AnalogMeasurer):
    peak_to_peak = Measure("Peak-to-Peak", type=float,
                           description="Peak to Peak", units="V")

    minimum = Measure("Min", type=float, description="Minimum", units="V")
    maximum = Measure("Max", type=float, description="Maximum", units="V")
    mean = Measure("Mean", type=float, description="Mean", units="V")
    mid = Measure("Mid", type=float, description="Midpoint", units="V")

    amplitude = Measure("Amplitude", type=float,
                        description="Amplitude", units="V")

    top = Measure("Top", type=float, description="High Level", units="V")
    base = Measure("Base", type=float, description="Low Level", units="V")
    rms = Measure("RMS", type=float,
                  description="Root Mean Square (RMS)", units="V")

    def measure_range(self, data: AnalogSpan):
        self.minimum.value = data.min
        self.maximum.value = data.max
        self.peak_to_peak.value = data.max - data.min
        mid = (data.max + data.min) / 2
        self.mid.value = mid

        # mean is the average of all samples.
        # TODO: replace with numpy
        if self.mean.enabled:
            self.mean.value = sum(data) / len(data)

        # amplitude is high - low.
        high = low = None

        if self.top.enabled or self.amplitude.enabled:
            high = compute_high(data, mid)
            self.top.value = high

        if self.base.enabled or self.amplitude.enabled:
            low = compute_low(data, mid)
            self.base.value = low

        if self.amplitude.enabled:
            self.amplitude.value = high - low

        if self.rms.enabled:
            self.rms.value = rms(data)


risetime_thresholds_90 = (0.1, 0.9)


class PulseMeasurer(AnalogMeasurer):
    rise_time = Measure("Rise Time", type=float,
                        description="Rise Time from 10% to 90%", units="s")
    fall_time = Measure("Fall Time", type=float,
                        description="Fall Time from 90% to 10%", units="s")
    positive_overshoot = Measure("+Overshoot", type=float,
                                 description="Positive Overshoot", units="V")
    negative_overshoot = Measure("-Overshoot", type=float,
                                 description="Negative Overshoot", units="V")

    def measure_range(self, data: AnalogSpan):
        mid = (data.max + data.min) / 2
        high = low = None

        if self.rise_time.enabled or self.fall_time.enabled or self.positive_overshoot.enabled:
            high = compute_high(data, mid)

        if self.rise_time.enabled or self.fall_time.enabled or self.negative_overshoot.enabled:
            low = compute_low(data, mid)

        if self.rise_time.enabled:
            low_threshold = (high - low) * risetime_thresholds_90[0] + low
            high_threshold = (high - low) * risetime_thresholds_90[1] + low

            rise_time, error = self.find_rise_fall_time(
                low_threshold, high_threshold, data)

            if error:
                self.rise_time.error = error
            elif rise_time:
                self.rise_time.value = rise_time

        if self.fall_time.enabled:
            low_threshold = (high - low) * risetime_thresholds_90[0] + low
            high_threshold = (high - low) * risetime_thresholds_90[1] + low

            fall_time, error = self.find_rise_fall_time(
                high_threshold, low_threshold, data)

            if error:
                self.fall_time.error = error
            elif fall_time:
                self.fall_time.value = fall_time

        if self.positive_overshoot.enabled:
            self.positive_overshoot.value = data.max - high

        if self.negative_overshoot.enabled:
            self.negative_overshoot.value = low - data.min

    def find_rise_fall_time(self, first_threshold, second_threshold, data: AnalogSpan):
        find_start = find_crossing = None
        if second_threshold >= first_threshold:
            find_start = data.find_lt
            find_crossing = data.find_ge
        else:
            find_start = data.find_gt
            find_crossing = data.find_le

        # find the first point that's below the start of the rising edge or above the start of the falling edge, to set the search origin.
        search_start = find_start(first_threshold)
        if search_start is None:
            return (None, "Edge Not Found")

        # find the first crossing of the first threshold
        first_crossing = find_crossing(first_threshold, start=search_start)
        if first_crossing is None:
            return (None, "1st Edge Not Found")

        # then find from there the first crossing of the second threshold
        second_crossing = find_crossing(
            second_threshold, start=first_crossing)
        if second_crossing is None:
            return (None, "2nd Edge Not Found")

        rise_time = float(data.sample_index_to_time(
            second_crossing) - data.sample_index_to_time(first_crossing))

        return(rise_time, None)


class TimeMeasurer(AnalogMeasurer):
    period = Measure("Period", type=float,
                     description="Period", units="s")
    frequency = Measure("Frequency", type=float,
                        description="Frequency", units="Hz")
    positive_width = Measure("Pos Width", type=float,
                             description="Positive Pulse Width", units="s")
    negative_width = Measure("Neg Width", type=float,
                             description="Negative Pulse Width", units="s")
    positive_duty = Measure("Pos Duty", type=float,
                            description="Positive Duty Cycle", units="%")
    negative_duty = Measure("Neg Duty", type=float,
                            description="Negative Duty Cycle", units="%")
    cycle_rms = Measure("Cycle RMS", type=float,
                        description="Cycle Root Mean Square (RMS)", units="V")

    def measure_range(self, data: AnalogSpan):
        start_index = int(len(data) / 2)

        mid = (data.max + data.min) / 2

        left_crossing = right_crossing = extra_left_crossing = extra_right_crossing = None

        # if the start location is above the mid level, we will find 2 edges to the right.
        # otherwise, we will find 2 edges to the left.
        # TODO: support a fallback, where if a cycle isn't found on one side, we should search the otherside.
        is_start_above_mid = data[start_index] >= mid

        left_func = data.rfind_lt if is_start_above_mid else data.rfind_ge
        right_func = data.find_lt if is_start_above_mid else data.find_ge
        extra_left_func = data.rfind_ge if is_start_above_mid else data.rfind_lt
        extra_right_func = data.find_ge if is_start_above_mid else data.find_lt
        left_crossing = left_func(mid, end=start_index)
        right_crossing = right_func(mid, start=start_index)

        if left_crossing is not None and right_crossing is not None:
            right_crossing_time = data.sample_index_to_time(right_crossing)
            left_crossing_time = data.sample_index_to_time(left_crossing)

            main_width = float(right_crossing_time - left_crossing_time)

            if is_start_above_mid:
                self.positive_width.value = main_width
            else:
                self.negative_width.value = main_width

            extra_left_crossing = extra_left_func(mid, end=left_crossing)
            extra_right_crossing = extra_right_func(mid, start=right_crossing)

            extra_crossing_origin = right_crossing if is_start_above_mid else left_crossing

            if extra_left_crossing is not None or extra_right_crossing is not None:
                # use the right side IF we started above the cycle, AND the right side is valid, OR
                # IF we started below the cycle, BUT the left side is not valid
                use_right_side = (is_start_above_mid and extra_right_crossing is not None) or (
                    not is_start_above_mid and extra_left_crossing is None)

                extra_crossing_time = data.sample_index_to_time(
                    extra_right_crossing) if use_right_side else data.sample_index_to_time(extra_left_crossing)
                first_crossing = left_crossing if use_right_side else extra_left_crossing
                last_crossing = extra_right_crossing if use_right_side else right_crossing
                first_crossing_time = data.sample_index_to_time(first_crossing)
                last_crossing_time = data.sample_index_to_time(last_crossing)

                extra_width = float((
                    extra_crossing_time - right_crossing_time) if use_right_side else (left_crossing_time - extra_crossing_time))

                if is_start_above_mid:
                    self.negative_width.value = extra_width
                else:
                    self.positive_width.value = extra_width

                period = float(last_crossing_time - first_crossing_time)
                positive_width = main_width if is_start_above_mid else extra_width
                negative_width = extra_width if is_start_above_mid else main_width

                frequency = 1 / period
                positive_duty = positive_width / period * 100
                negative_duty = negative_width / period * 100

                self.period.value = period
                self.frequency.value = frequency
                self.positive_duty.value = positive_duty
                self.negative_duty.value = negative_duty

                if self.cycle_rms.enabled:
                    print(first_crossing, last_crossing)
                    slice = data[first_crossing:last_crossing]
                    self.cycle_rms.value = rms(slice)
            else:
                # all 3 point measurements failed.
                error_string = "3rd Crossing Not Found"
                self.period.error = error_string
                self.frequency.error = error_string
                self.positive_duty.error = error_string
                self.negative_duty.error = error_string
                self.cycle_rms.error = error_string
                if is_start_above_mid:
                    self.negative_width.error = error_string
                else:
                    self.positive_width.error = error_string
        else:
            # all measurements failed
            error_string = "Crossings Not Found"
            self.period.error = error_string
            self.frequency.error = error_string
            self.positive_width.error = error_string
            self.negative_width.error = error_string
            self.positive_duty.error = error_string
            self.negative_duty.error = error_string
            self.cycle_rms.error = error_string
