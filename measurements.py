from saleae.measurements import AnalogMeasurer, Measure
from saleae.data import AnalogSpan
from statistics import mode


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

        # amplitude is high - low. How should those be defined?
        # TODO: consider computing high/low using a different method.
        high = low = None
        if self.top.enabled or self.amplitude.enabled:
            high = mode(x for x in data if x >= mid)
            self.top.value = high

        if self.base.enabled or self.amplitude.enabled:
            low = mode(x for x in data if x < mid)
            self.base.value = low

        if self.amplitude.enabled:
            self.amplitude.value = high - low


class PulseMeasurer(AnalogMeasurer):
    rise_time = Measure("Rise Time", type=float,
                        description="Rise Time from 10% to 90%", units="s")
    fall_time = Measure("Fall Time", type=float,
                        description="Fall Time from 90% to 91%", units="s")
    positive_overshoot = Measure("+Overshoot", type=float,
                                 description="Positive Overshoot", units="V")
    negative_overshoot = Measure("-Overshoot", type=float,
                                 description="Negative Overshoot", units="V")

    def measure_range(self, data: AnalogSpan):
        mid = (data.max + data.min) / 2
        high = low = None

        if self.rise_time.enabled or self.fall_time.enabled or self.positive_overshoot.enabled:
            high = mode(x for x in data if x >= mid)

        if self.rise_time.enabled or self.fall_time.enabled or self.negative_overshoot.enabled:
            low = mode(x for x in data if x < mid)

        if self.rise_time.enabled:
            low_threshold = (high - low) * 0.1 + low
            high_threshold = (high - low) * 0.9 + low

            rise_time, error = self.find_rise_fall_time(
                low_threshold, high_threshold, data)

            if error:
                self.rise_time.error = error
            elif rise_time:
                self.rise_time.value = rise_time

        if self.fall_time.enabled:
            low_threshold = (high - low) * 0.1 + low
            high_threshold = (high - low) * 0.9 + low

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
        error_response = (None, "Not Found")
        if second_threshold >= first_threshold:
            find_start = data.find_lt
            find_crossing = data.find_ge
        else:
            find_start = data.find_gt
            find_crossing = data.find_le

        # find the first point that's below the start of the rising edge or above the start of the falling edge, to set the search origin.
        search_start = find_start(first_threshold)
        if search_start is None:
            return error_response

        # find the first crossing of the first threshold
        first_crossing = find_crossing(first_threshold, start=search_start)
        if first_crossing is None:
            return error_response

        # then find from there the first crossing of the second threshold
        second_crossing = find_crossing(
            second_threshold, start=first_crossing)
        if second_crossing is None:
            return error_response

        # TODO: implement sample_index_to_time
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

    def measure_range(self, data: AnalogSpan):
        start_index = int(len(data) / 2)

        mid = (data.max + data.min) / 2

        left_crossing = right_crossing = extra_crossing = None

        # if the start location is above the mid level, we will find 2 edges to the right.
        # otherwise, we will find 2 edges to the left.
        is_start_above_mid = data[start_index] >= mid

        left_func = data.rfind_lt if is_start_above_mid else data.rfind_ge
        right_func = data.find_lt if is_start_above_mid else data.find_ge
        extra_func = data.find_ge if is_start_above_mid else data.rfind_lt
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

            extra_crossing_origin = right_crossing if is_start_above_mid else left_crossing
            if is_start_above_mid:
                extra_crossing = extra_func(mid, start=extra_crossing_origin)
            else:
                extra_crossing = extra_func(mid, end=extra_crossing_origin)
            if extra_crossing is not None:
                extra_crossing_time = data.sample_index_to_time(extra_crossing)

                first_crossing_time = left_crossing_time if is_start_above_mid else extra_crossing_time
                last_crossing_time = extra_crossing_time if is_start_above_mid else right_crossing_time

                extra_width = float((
                    extra_crossing_time - right_crossing_time) if is_start_above_mid else (left_crossing_time - extra_crossing_time))

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
        pass
