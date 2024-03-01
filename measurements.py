import math
from statistics import mode
from typing import Optional, Tuple

from saleae.data import AnalogSpan, GraphTimeDelta
from saleae.measurements import (AnalogMeasurer, Annotation, HorizontalRule,
                                 Measure, VerticalRule)

CLIPPED_ERROR = 'Clipped'

def rms(data: AnalogSpan):
    # TODO: numpy accelerate
    sum_of_squares = sum(x**2 for x in data)
    rms = math.sqrt(sum_of_squares / len(data))
    return rms


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

    max_annotation = Annotation(measures=[peak_to_peak, maximum])
    min_annotation = Annotation(measures=[peak_to_peak, minimum])

    mean_annotation = Annotation(measures=[mean])
    mid_annotation = Annotation(measures=[mid])
    top_annotation = Annotation(measures=[top, amplitude])
    base_annotation = Annotation(measures=[base, amplitude])

    def measure_range(self, data: AnalogSpan):
        if data.clipped:
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
            mean = sum(data) / len(data)
            self.mean.value = mean
            self.mean_annotation.value = HorizontalRule(value=mean)

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
                        description="Rise Time from 10% to 90%", units="s")
    fall_time = Measure("Fall Time", type=float,
                        description="Fall Time from 90% to 10%", units="s")
    positive_overshoot = Measure("+Overshoot", type=float,
                                 description="Positive Overshoot", units="V")
    negative_overshoot = Measure("-Overshoot", type=float,
                                 description="Negative Overshoot", units="V")

    rise_time_annotation = Annotation(measures=[rise_time])
    fall_time_annotation = Annotation(measures=[fall_time])
    rise_fall_threshold_annotation = Annotation(
        measures=[rise_time, fall_time])
    positive_overshoot_annotation = Annotation(measures=[positive_overshoot])
    negative_overshoot_annotation = Annotation(measures=[negative_overshoot])

    def measure_range(self, data: AnalogSpan):
        if data.clipped:
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


FREQUENCY_HYSTERESIS_PCT = 0.1

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

    positive_side_annotation = Annotation(
        measures=[period, frequency, positive_width, positive_duty, negative_duty, cycle_rms])
    negative_side_annotation = Annotation(
        measures=[period, frequency, negative_width, positive_duty, negative_duty, cycle_rms])
    center_annotation = Annotation(
        measures=[positive_width, negative_width, positive_duty, negative_duty])

    def measure_range(self, data: AnalogSpan):
        if data.clipped:
            self.positive_width.error = CLIPPED_ERROR
            self.negative_width.error = CLIPPED_ERROR
            self.positive_duty.error = CLIPPED_ERROR
            self.negative_duty.error = CLIPPED_ERROR
            self.cycle_rms.error = CLIPPED_ERROR

        start_index = int(len(data) / 2)

        mid = (data.max + data.min) / 2
        hysteresis = (data.max - data.min) * FREQUENCY_HYSTERESIS_PCT

        start_value = data[start_index]
        if (abs(start_value - mid) < hysteresis):
            lt_index = data.find_lt(mid - hysteresis, start=start_index)
            gt_index = data.find_gt(mid - hysteresis, start=start_index)
            start_index = min(lt_index, gt_index)


        left_crossing = right_crossing = extra_left_crossing = extra_right_crossing = None

        # if the start location is above the mid level, we will find 2 edges to the right.
        # otherwise, we will find 2 edges to the left.
        # TODO: support a fallback, where if a cycle isn't found on one side, we should search the otherside.
        is_start_above_mid = data[start_index] >= mid

        # We use the mid value to determine where we are in the period, and whether we should search above or below for crossings.
        left_func = data.rfind_lt if is_start_above_mid else data.rfind_ge
        right_func = data.find_lt if is_start_above_mid else data.find_ge
        extra_left_func = data.rfind_ge if is_start_above_mid else data.rfind_lt
        extra_right_func = data.find_ge if is_start_above_mid else data.find_lt
        extra_hysteresis = -hysteresis if is_start_above_mid else hysteresis
        left_crossing = left_func(mid, end=start_index)
        right_crossing = right_func(mid, start=start_index)

        if left_crossing is not None and right_crossing is not None:
            right_crossing_time = data.sample_index_to_time(right_crossing)
            left_crossing_time = data.sample_index_to_time(left_crossing)

            main_width = float(right_crossing_time - left_crossing_time)

            # Move past the hysteresis point
            extra_left_crossing_start = left_func(mid + extra_hysteresis, end=left_crossing)
            extra_left_crossing = extra_left_func(mid, end=extra_left_crossing_start)
            extra_right_crossing_start = right_func(mid + extra_hysteresis, start=right_crossing)
            extra_right_crossing = extra_right_func(mid, start=extra_right_crossing_start)

            if extra_left_crossing is not None or extra_right_crossing is not None:
                # use the right side IF we started above the cycle, AND the right side is valid, OR
                # IF we started below the cycle, BUT the left side is not valid
                use_right_side = (is_start_above_mid and extra_right_crossing is not None) or (
                    not is_start_above_mid and extra_left_crossing is None)

                extra_crossing_time = data.sample_index_to_time(
                    extra_right_crossing) if use_right_side else data.sample_index_to_time(extra_left_crossing)
                first_crossing = left_crossing if use_right_side else extra_left_crossing
                last_crossing = extra_right_crossing if use_right_side else right_crossing
                center_crossing = right_crossing if use_right_side else left_crossing
                first_crossing_time = data.sample_index_to_time(first_crossing)
                last_crossing_time = data.sample_index_to_time(last_crossing)
                center_crossing_time = data.sample_index_to_time(
                    center_crossing)


                self.center_annotation.value = VerticalRule(time=center_crossing_time)

                # left side is positive IF: (A) is_start_above_mid & use_right_side or (B) !is_start_above_mid & !use_right_side
                first_time_is_positive = (is_start_above_mid and use_right_side) or (
                    not is_start_above_mid and not use_right_side)
                self.positive_side_annotation.value = VerticalRule(
                    time=first_crossing_time if first_time_is_positive else last_crossing_time)
                self.negative_side_annotation.value = VerticalRule(
                    time=last_crossing_time if first_time_is_positive else first_crossing_time)

                extra_width = float((
                    extra_crossing_time - right_crossing_time) if use_right_side else (left_crossing_time - extra_crossing_time))


                period = float(last_crossing_time - first_crossing_time)
                positive_width = main_width if is_start_above_mid else extra_width
                negative_width = extra_width if is_start_above_mid else main_width

                self.positive_width.value = positive_width
                self.negative_width.value = negative_width

                frequency = 1 / period
                positive_duty = positive_width / period * 100
                negative_duty = negative_width / period * 100

                self.period.value = period
                self.frequency.value = frequency
                self.positive_duty.value = positive_duty
                self.negative_duty.value = negative_duty

                if self.cycle_rms.enabled:
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

                    self.positive_side_annotation.value = VerticalRule(
                        time=left_crossing_time)
                    self.center_annotation.value = VerticalRule(
                        time=right_crossing_time)
                else:
                    self.positive_width.error = error_string

                    self.negative_side_annotation.value = VerticalRule(
                        time=left_crossing_time)
                    self.center_annotation.value = VerticalRule(
                        time=right_crossing_time)
        else:
            # all measurements failed
            error_string = "Crossings Not Found"
            self.period.error = error_string
            self.frequency.error = error_string
            if not data.clipped:
                self.positive_width.error = error_string
                self.negative_width.error = error_string
                self.positive_duty.error = error_string
                self.negative_duty.error = error_string
                self.cycle_rms.error = error_string
