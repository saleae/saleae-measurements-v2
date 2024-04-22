import math
from statistics import mode
from typing import List, Optional, Tuple

import numpy as np

from saleae.data import AnalogSpan, GraphTimeDelta
from saleae.measurements import (AnalogMeasurer, Annotation, HorizontalRule,
                                 Measure, VerticalRule)

CLIPPED_ERROR = 'Clipped'
SIGNAL_TOO_SMALL_ERROR = 'Amplitude too small'

def is_clipped(data: AnalogSpan):
    """
    Return True if the data is at the ADC rails.
    """
    return data.min == data._adc_to_voltage(data.acquisition_info.min_adc_code) \
        or data.max == data._adc_to_voltage(data.acquisition_info.max_adc_code)


def mean(data: AnalogSpan):
    total_sum = 0
    total_samples = 0

    for chunk in data.raw_chunks():
        total_sum += chunk._convert_sample(np.sum(chunk.raw_samples, dtype=np.int64)) + chunk.voltage_transform.offset * len(chunk.raw_samples)
        total_samples += len(chunk.raw_samples)

    return total_sum / total_samples


def rms(data: AnalogSpan):
    total_sum = 0
    total_samples = 0

    for chunk in data.raw_chunks():
        total_sum += np.sum(chunk.samples ** 2)
        total_samples += len(chunk.raw_samples)

    return math.sqrt(total_sum / total_samples)


def histogram_mode(data: AnalogSpan, filter):
    hist, bins = data.histogram()
    _count, index = max((count, index) for index, count in enumerate(
        hist) if filter((bins[index] + bins[index + 1]) / 2))
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


def find_three_closest(nums: List[Tuple[int, bool]], target: int):
    # If the array has 3 or fewer elements, return the entire array
    if len(nums) <= 3:
        return nums

    # Calculate the difference from the target for each number
    diff_list = [abs(num[0] - target) for num in nums] #Get the indices of the three smallest differences
    # This maintains the original order of the elements
    idx_three_closest = sorted(
        range(len(diff_list)), key=lambda i: diff_list[i])[:3]

    # Select the numbers at these indices, maintaining original order
    closest_three = [nums[i] for i in sorted(idx_three_closest)]

    return closest_three


def find_starting_location(data, mid, hysteresis) -> Optional[int]:
    """Find a starting location that is outside of the hysteresis band."""

    start_index = int(len(data) / 2)
    start_value = data[start_index]
    if (abs(start_value - mid) < hysteresis):
        right_lt_index = data.find_lt(mid - hysteresis, start=start_index)
        right_gt_index = data.find_gt(mid + hysteresis, start=start_index)
        left_lt_index = data.rfind_lt(mid - hysteresis, end=start_index)
        left_gt_index = data.rfind_gt(mid + hysteresis, end=start_index)
        # find the nearest sample index that is outside of the hysteresis range.
        indexes = [right_lt_index, right_gt_index,
                   left_lt_index, left_gt_index]
        valid_indexes = [v for v in indexes if v is not None]
        if not valid_indexes:
            # in this case, no sample in the data is outside of the hysteresis zone.
            # since hysteresis is always smaller than the amplitude, this should only happen for a constant value signal.
            return None
        closest_index = min(
            valid_indexes, key=lambda x: abs(x - start_index))

        start_index = closest_index
    return start_index


def find_cycle(data, start_index, mid, hysteresis) -> Tuple[List[int], bool]:
    """
    Find the 2 or 3 closest crossing from `start_index`.
    A positive crossing must go from `mid - hysteresis` to `> mid`.
    A negative crossing must go from `mid + hysteresis` to `< mid`.

    Returns a tuple of the 2 or 3 closest crossings, and a boolean indicating if the first cycle is positive or negative.

    """
    # find up to 6 crossings. store the location, and if the signal on the right is above the threshold.
    crossings: List[Tuple[int, bool]] = []

    is_start_above_mid = data[start_index] >= mid

    # We use the mid value to determine where we are in the period, and whether we should search above or below for crossings.
    leading_left_func = data.rfind_lt if is_start_above_mid else data.rfind_ge
    leading_right_func = data.find_lt if is_start_above_mid else data.find_ge
    leading_hysteresis = -hysteresis if is_start_above_mid else hysteresis

    trailing_left_func = data.rfind_ge if is_start_above_mid else data.rfind_lt
    trailing_right_func = data.find_ge if is_start_above_mid else data.find_lt
    trailing_hysteresis = hysteresis if is_start_above_mid else -hysteresis

    # find up to 3 crossings in each direction.
    l1 = leading_left_func(mid, end=start_index)

    if l1 is not None:
        crossings.insert(0, (l1, is_start_above_mid))
        # advance to hysteresis level.
        l1 = leading_left_func(mid + leading_hysteresis, end=l1)
        if l1 is not None:
            l2 = trailing_left_func(mid, end=l1)
            if l2 is not None:
                crossings.insert(0, (l2, not is_start_above_mid))
                # advance to hysteresis level.
                l2 = trailing_left_func(mid + trailing_hysteresis, end=l2)
                if l2 is not None:
                    l3 = leading_left_func(mid, end=l2)
                    if l3 is not None:
                        crossings.insert(0, (l3, is_start_above_mid))

    r1 = leading_right_func(mid, start=start_index)
    if r1 is not None:
        crossings.append((r1, not is_start_above_mid))
        # advance to hysteresis level.
        r1 = leading_right_func(mid + leading_hysteresis, start=r1)
        if r1 is not None:
            r2 = trailing_right_func(mid, start=r1)
            if r2 is not None:
                crossings.append((r2, is_start_above_mid))
                # advance to hysteresis level.
                r2 = trailing_right_func(mid + trailing_hysteresis, start=r2)
                if r2 is not None:
                    r3 = leading_right_func(mid, start=r2)
                    if r3 is not None:
                        crossings.append((r3, not is_start_above_mid))

    # we want to take the 3 crossings closest to the center.
    closest_crossings = find_three_closest(crossings, start_index)
    return ([x[0] for x in closest_crossings], closest_crossings[0][1])


class VoltageMeasurer(AnalogMeasurer):
    peak_to_peak = Measure("Peak-to-Peak", type=float,
                           description="Voltage difference between maximum and minimum points.", units="V")
    minimum = Measure(
        "Min", type=float, description="Lowest voltage value of the waveform.", units="V")
    maximum = Measure(
        "Max", type=float, description="Highest voltage value of the waveform.", units="V")
    mean = Measure("Mean", type=float,
                   description="Average voltage level over the waveform.", units="V")
    mid = Measure("Mid", type=float,
                  description="Midpoint voltage between maximum and minimum.", units="V")

    amplitude = Measure("Amplitude", type=float,
                        description="Voltage difference between base and top values.", units="V")

    top = Measure("Top", type=float,
                  description="Most prevalent voltage above the midpoint of the waveform.", units="V")
    base = Measure("Base", type=float,
                   description="Most prevalent voltage below the midpoint of the waveform.", units="V")
    rms = Measure("RMS", type=float,
                  description="Root Mean Square (RMS) taken over the full waveform, ground referenced.", units="V")

    max_annotation = Annotation(measures=[peak_to_peak, maximum])
    min_annotation = Annotation(measures=[peak_to_peak, minimum])

    mean_annotation = Annotation(measures=[mean])
    mid_annotation = Annotation(measures=[mid])
    top_annotation = Annotation(measures=[top, amplitude])
    base_annotation = Annotation(measures=[base, amplitude])

    def measure_range(self, data: AnalogSpan):
        if is_clipped(data):
            self.peak_to_peak.error = CLIPPED_ERROR
            self.minimum.error = CLIPPED_ERROR
            self.maximum.error = CLIPPED_ERROR
            self.mean.error = CLIPPED_ERROR
            self.mid.error = CLIPPED_ERROR
            self.amplitude.error = CLIPPED_ERROR
            self.top.error = CLIPPED_ERROR
            self.base.error = CLIPPED_ERROR
            self.rms.error = CLIPPED_ERROR
            return

        self.minimum.value = data.min
        self.maximum.value = data.max
        self.peak_to_peak.value = data.max - data.min
        mid = (data.max + data.min) / 2
        self.mid.value = mid

        self.max_annotation.value = HorizontalRule(value=data.max)
        self.min_annotation.value = HorizontalRule(value=data.min)
        self.mid_annotation.value = HorizontalRule(value=mid)

        # mean is the average of all samples.
        # TODO: replace with numpy
        if self.mean.enabled:
            mean_value = mean(data)
            self.mean.value = mean_value
            self.mean_annotation.value = HorizontalRule(value=mean_value)

        # amplitude is high - low.
        high = low = None

        if self.top.enabled or self.amplitude.enabled:
            high = compute_high(data, mid)
            self.top.value = high
            self.top_annotation.value = HorizontalRule(value=high)

        if self.base.enabled or self.amplitude.enabled:
            low = compute_low(data, mid)
            self.base.value = low
            self.base_annotation.value = HorizontalRule(value=low)

        if self.amplitude.enabled:
            self.amplitude.value = high - low

        if self.rms.enabled:
            self.rms.value = rms(data)


risetime_thresholds_90 = (0.1, 0.9)


class PulseMeasurer(AnalogMeasurer):
    rise_time = Measure("Rise Time", type=float,
                        description="Time waveform takes to rise from 10% to 90%.", units="s")
    fall_time = Measure("Fall Time", type=float,
                        description="Time waveform takes to fall from 90% to 10%.", units="s")
    positive_overshoot = Measure("+Overshoot", type=float,
                                 description="Magnitude from the top voltage to the maximum voltage.", units="V")
    negative_overshoot = Measure("-Overshoot", type=float,
                                 description="Magnitude from the base voltage to the minimum voltage.", units="V")

    rise_time_annotation = Annotation(measures=[rise_time])
    fall_time_annotation = Annotation(measures=[fall_time])
    rise_fall_threshold_annotation = Annotation(
        measures=[rise_time, fall_time])
    positive_overshoot_annotation = Annotation(measures=[positive_overshoot])
    negative_overshoot_annotation = Annotation(measures=[negative_overshoot])

    def measure_range(self, data: AnalogSpan):
        if is_clipped(data):
            self.rise_time.error = CLIPPED_ERROR
            self.fall_time.error = CLIPPED_ERROR
            self.positive_overshoot.error = CLIPPED_ERROR
            self.negative_overshoot.error = CLIPPED_ERROR
            return

        mid = (data.max + data.min) / 2
        high = low = None

        if self.rise_time.enabled or self.fall_time.enabled or self.positive_overshoot.enabled:
            high = compute_high(data, mid)

        if self.rise_time.enabled or self.fall_time.enabled or self.negative_overshoot.enabled:
            low = compute_low(data, mid)

        low_threshold = high_threshold = None
        if self.rise_time.enabled or self.fall_time.enabled:
            low_threshold = (high - low) * risetime_thresholds_90[0] + low
            high_threshold = (high - low) * risetime_thresholds_90[1] + low
            self.rise_fall_threshold_annotation.value = [HorizontalRule(
                value=low_threshold), HorizontalRule(value=high_threshold)]

        if self.rise_time.enabled:
            rise_time, first_crossing_time, second_crossing_time, error = self.find_rise_fall_time(
                low_threshold, high_threshold, data)

            if error:
                self.rise_time.error = error
            elif rise_time:
                self.rise_time.value = rise_time
                self.rise_time_annotation.value = [VerticalRule(
                    time=first_crossing_time), VerticalRule(time=second_crossing_time)]

        if self.fall_time.enabled:
            fall_time, first_crossing_time, second_crossing_time, error = self.find_rise_fall_time(
                high_threshold, low_threshold, data)

            if error:
                self.fall_time.error = error
            elif fall_time:
                self.fall_time.value = fall_time
                self.fall_time_annotation.value = [VerticalRule(
                    time=first_crossing_time), VerticalRule(time=second_crossing_time)]

        if self.positive_overshoot.enabled:
            self.positive_overshoot.value = data.max - high
            self.positive_overshoot_annotation.value = [HorizontalRule(
                value=high), HorizontalRule(value=data.max)]

        if self.negative_overshoot.enabled:
            self.negative_overshoot.value = low - data.min
            self.negative_overshoot_annotation.value = [HorizontalRule(
                value=low), HorizontalRule(value=data.min)]

    def find_rise_fall_time_directional(self, first_threshold: float, second_threshold: float, data: AnalogSpan, start: int, reverse: bool) -> Tuple[Optional[float], Optional[GraphTimeDelta], Optional[GraphTimeDelta], Optional[str]]:
        # This measures the rise or fall time, starting the search at the provided index and moving in the specified direction.
        # if the search start is between the thresholds, we will back up the start point to the last point that satisfied the first threshold, if and only if the start location is within the correct edge direction.
        # when running in reverse, we will find the second threshold first. This allows you to call this function in both directions with the same thresholds.
        find_start = find_crossing = find_backtrack_crossing = find_backtrack_start = None
        # start_key and end_key are used to specify the direction of the search when calling find_* or rfind_* functions.
        # when searching backwards, the search begins at the specified end sample, and continues toward the start of the data.
        start_key = 'start'
        end_key = 'end'
        if reverse:
            start_key = 'end'
            end_key = 'start'

        if second_threshold >= first_threshold:
            find_start = data.find_lt
            find_crossing = data.find_ge
            find_backtrack_crossing = data.rfind_le
            find_backtrack_start = data.rfind_lt
            if reverse:
                temp_threshold = first_threshold
                first_threshold = second_threshold
                second_threshold = temp_threshold
                find_start = data.rfind_gt
                find_crossing = data.rfind_le
                find_backtrack_crossing = data.find_ge
                find_backtrack_start = data.find_gt
        else:
            find_start = data.find_gt
            find_crossing = data.find_le
            find_backtrack_crossing = data.rfind_ge
            find_backtrack_start = data.rfind_gt
            if reverse:
                temp_threshold = first_threshold
                first_threshold = second_threshold
                second_threshold = temp_threshold
                find_start = data.rfind_lt
                find_crossing = data.rfind_ge
                find_backtrack_crossing = data.find_le
                find_backtrack_start = data.find_lt

        # check to see if the start point is between the thresholds.
        high_threshold = max(first_threshold, second_threshold)
        low_threshold = min(first_threshold, second_threshold)
        if low_threshold <= data[start] and data[start] <= high_threshold:
            # We could be inside a rising or falling edge. We only care if we're inside the edge we're currently looking for.
            next_start = find_start(
                first_threshold, **{start_key: start})
            next_end = find_crossing(
                second_threshold, **{start_key: start})
            # if ONLY next_edge is found, or if both are found, but next end is before next start, then we have a a logical end point and we should back-up if possible.
            if next_end is not None or (next_start is not None and next_end is not None and next_end < next_start):
                # we're inside the edge. We need to back the start point up.
                backtracked_start = find_backtrack_start(
                    first_threshold, **{end_key: start})
                # only back-up if there actually is a crossing to backup to.
                if backtracked_start is not None:
                    start = backtracked_start

        # find the first point that's below the start of the rising edge or above the start of the falling edge, to set the search origin.
        search_start = find_start(first_threshold, **{start_key: start})
        if search_start is None:
            return (None, None, None, "Edge Not Found")

        # find the first crossing of the first threshold
        first_crossing = find_crossing(
            first_threshold, **{start_key: search_start})
        if first_crossing is None:
            return (None, None, None, "1st Edge Not Found")

        # then find from there the first crossing of the second threshold
        second_crossing = find_crossing(
            second_threshold, **{start_key: first_crossing})
        if second_crossing is None:
            return (None, None, None, "2nd Edge Not Found")

        # back up to the last sample that satisfied the first threshold, and use that as the first_crossing instead. This rejects noise.
        first_crossing_backtrack = find_backtrack_crossing(
            first_threshold, **{end_key: second_crossing})
        if first_crossing_backtrack is None:
            return (None, None, None, "Error")

        first_crossing_time = data.sample_index_to_time(
            first_crossing_backtrack)
        second_crossing_time = data.sample_index_to_time(second_crossing)

        rise_time = abs(float(second_crossing_time - first_crossing_time))

        if reverse:
            return (rise_time, second_crossing_time, first_crossing_time, None)

        return (rise_time, first_crossing_time, second_crossing_time, None)

    def find_rise_fall_time(self, first_threshold: float, second_threshold: float, data: AnalogSpan) -> Tuple[Optional[float], Optional[GraphTimeDelta], Optional[GraphTimeDelta], Optional[str]]:
        # use first_threshold > second_threshold to find the fall time. Otherwise, find the rise time.
        center_index = int(len(data) / 2)
        center_time = data.sample_index_to_time(center_index)
        # check for the rise/fall time, starting to the right of the center of the data.
        right_rise_time, right_first_crossing_time, right_second_crossing_time, right_error = self.find_rise_fall_time_directional(
            first_threshold, second_threshold, data, center_index, False)

        left_rise_time, left_first_crossing_time, left_second_crossing_time, left_error = self.find_rise_fall_time_directional(
            first_threshold, second_threshold, data, center_index, True)

        # if we found an event on both sides of the center, use the one with the nearest crossing time..
        if right_rise_time and left_rise_time and left_second_crossing_time and right_first_crossing_time:
            # find which side has the nearest edge.
            left_side_distance = abs(
                float(center_time - left_second_crossing_time))
            right_side_distance = abs(
                float(right_first_crossing_time - center_time))
            if left_side_distance < right_side_distance:
                return (left_rise_time, left_first_crossing_time, left_second_crossing_time, left_error)
            else:
                return (right_rise_time, right_first_crossing_time, right_second_crossing_time, right_error)

        elif right_rise_time:
            return (right_rise_time, right_first_crossing_time, right_second_crossing_time, right_error)
        elif left_rise_time:
            return (left_rise_time, left_first_crossing_time, left_second_crossing_time, left_error)
        else:
            error = right_error if right_error else left_error
            return (None, None, None, error if error else "Edge Not Found")


FREQUENCY_HYSTERESIS_PCT = 0.2


class TimeMeasurer(AnalogMeasurer):
    period = Measure("Period", type=float,
                     description="Duration of one complete waveform cycle.", units="s")
    frequency = Measure("Frequency", type=float,
                        description="Cycles completed per second, computed from the period of a single cycle.", units="Hz")
    positive_width = Measure("Pos Width", type=float,
                             description="Time waveform spends above the midpoint in a cycle.", units="s")
    negative_width = Measure("Neg Width", type=float,
                             description="Time waveform sends below the midpoint in a cycle.", units="s")
    positive_duty = Measure("Pos Duty", type=float,
                            description="Percentage of cycle waveform is above the midpoint.", units="%")
    negative_duty = Measure("Neg Duty", type=float,
                            description="Percentage of cycle waveform is below the midpoint.", units="%")
    cycle_rms = Measure("Cycle RMS", type=float,
                        description="RMS value for a single waveform cycle, ground referenced.", units="V")

    positive_side_annotation = Annotation(
        measures=[period, frequency, positive_width, positive_duty, negative_duty, cycle_rms])
    negative_side_annotation = Annotation(
        measures=[period, frequency, negative_width, positive_duty, negative_duty, cycle_rms])
    center_annotation = Annotation(
        measures=[positive_width, negative_width, positive_duty, negative_duty])

    def measure_range(self, data: AnalogSpan):
        if is_clipped(data):
            self.positive_width.error = CLIPPED_ERROR
            self.negative_width.error = CLIPPED_ERROR
            self.positive_duty.error = CLIPPED_ERROR
            self.negative_duty.error = CLIPPED_ERROR
            self.cycle_rms.error = CLIPPED_ERROR

        mid = (data.max + data.min) / 2
        hysteresis = (data.max - data.min) * FREQUENCY_HYSTERESIS_PCT

        start_index = find_starting_location(data, mid, hysteresis)

        (closest_crossings, first_cycle_positive) = find_cycle(
            data, start_index, mid, hysteresis)

        if len(closest_crossings) <= 1:
            # we did not get enough crossings for any measurement
            # all measurements failed
            error_string = "Crossings Not Found"
            self.period.error = error_string
            self.frequency.error = error_string
            if not is_clipped(data):
                self.positive_width.error = error_string
                self.negative_width.error = error_string
                self.positive_duty.error = error_string
                self.negative_duty.error = error_string
                self.cycle_rms.error = error_string

        if len(closest_crossings) == 2:
            # we have a single pulse. There is no complete frequency. we should return positive or negative width, based on state between points.
            left_crossing = closest_crossings[0]
            right_crossing = closest_crossings[1]

            left_crossing_time = data.sample_index_to_time(left_crossing)
            right_crossing_time = data.sample_index_to_time(right_crossing)
            if first_cycle_positive:
                self.positive_side_annotation.value = VerticalRule(
                    time=left_crossing_time)
                self.center_annotation.value = VerticalRule(
                    time=right_crossing_time)
                self.positive_width.value = float(
                    right_crossing_time - left_crossing_time)
            else:
                self.negative_side_annotation.value = VerticalRule(
                    time=left_crossing_time)
                self.center_annotation.value = VerticalRule(
                    time=right_crossing_time)
                self.negative_width.value = float(
                    right_crossing_time - left_crossing_time)

            error_string = "3rd Crossing Not Found"
            self.period.error = error_string
            self.frequency.error = error_string
            if not is_clipped(data):
                if first_cycle_positive:
                    self.negative_width.error = error_string
                else:
                    self.positive_width.error = error_string
                self.positive_duty.error = error_string
                self.negative_duty.error = error_string
                self.cycle_rms.error = error_string

        if len(closest_crossings) == 3:
            # we have 3 crossings!
            left_crossing = closest_crossings[0]
            center_crossing = closest_crossings[1]
            right_crossing = closest_crossings[2]

            if (left_crossing >= center_crossing or center_crossing >= right_crossing):
                raise ValueError(
                    f"Crossings are not in order. This should not happen. crossings: {closest_crossings}")
            left_crossing_time = data.sample_index_to_time(left_crossing)
            right_crossing_time = data.sample_index_to_time(right_crossing)
            center_crossing_time = data.sample_index_to_time(center_crossing)

            self.center_annotation.value = VerticalRule(
                time=center_crossing_time)
            self.negative_side_annotation.value = VerticalRule(
                time=right_crossing_time if first_cycle_positive else left_crossing_time)
            self.positive_side_annotation.value = VerticalRule(
                time=left_crossing_time if first_cycle_positive else right_crossing_time)

            period = float(right_crossing_time - left_crossing_time)
            if period == 0:
                raise ValueError(
                    f"Period is zero. This should not happen. right: {right_crossing_time}, left: {left_crossing_time}. crossings: {closest_crossings}")
            frequency = 1 / period
            positive_width = abs(float(
                center_crossing_time - (left_crossing_time if first_cycle_positive else right_crossing_time)))
            negative_width = abs(float(
                center_crossing_time - (right_crossing_time if first_cycle_positive else left_crossing_time)))
            positive_duty = positive_width / period * 100
            negative_duty = negative_width / period * 100

            self.period.value = period
            self.frequency.value = frequency
            self.positive_duty.value = positive_duty
            self.negative_duty.value = negative_duty
            self.positive_width.value = positive_width
            self.negative_width.value = negative_width

            if self.cycle_rms.enabled:
                cycle_slice = data[left_crossing:right_crossing]
                self.cycle_rms.value = rms(cycle_slice)
