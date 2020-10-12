import numpy as np
import pandas as pd


def _apply_tpm(lanes_to_draw, raw_data):
    """Convert read counts in raw_data into TPMs - in situ"""
    for lane in lanes_to_draw:
        norm_factor = 1e6 / lane.get_aligned_read_count()
        raw_data[lane.name] = raw_data[lane.name] * norm_factor


class _Normalization:
    """Base class for all normalizations"""

    pass


class AsIs(_Normalization):
    name = "NormRaw"

    def get_dependencies(self, lanes_to_draw):
        return []

    def calc(self, lanes_to_draw, raw_data):
        return raw_data


class NormLaneTPM(_Normalization):
    """Normalize to TPM based on lane.get_aligned_read_count"""

    name = "Norm_Lane_TPM"

    def get_dependencies(self, lanes_to_draw):
        return [x.count_aligned_reads() for x in lanes_to_draw]

    def calc(self, lanes_to_draw, raw_data):
        _apply_tpm(lanes_to_draw, raw_data)
        return raw_data


class NormLaneTPMInterpolate(_Normalization):
    """Normalize to TPM based on lane.get_aligned_read_count, then reduce data by interpolation (for large regions)"""

    def __init__(self, samples_per_region=100):
        self.name = "Norm_Lane_TPM_interpolated_%i" % samples_per_region
        self.samples_per_region = samples_per_region

    def get_dependencies(self, lanes_to_draw):
        return [x.count_aligned_reads() for x in lanes_to_draw]

    def calc(self, lanes_to_draw, raw_data):
        for lane in lanes_to_draw:
            _apply_tpm(lanes_to_draw, raw_data)
            cv = raw_data[lane.name]
            new_rows = []
            for row_no in range(0, cv.shape[0]):
                row = cv[row_no]
                interp = np.interp(
                    [
                        len(row) / float(self.samples_per_region) * ii
                        for ii in range(0, self.samples_per_region)
                    ],
                    range(0, len(row)),
                    row,
                )
                new_rows.append(interp)
            raw_data[lane.name] = np.array(new_rows)
        return raw_data


class NormLaneMax(_Normalization):
    """Normalize to the maximum value of the regions in each lane"""

    def __init__(self):
        self.name = "NormLaneMax"

    def get_dependencies(self, lanes_to_draw):
        return [x.count_aligned_reads() for x in lanes_to_draw]

    def calc(self, lanes_to_draw, raw_data):
        for lane in lanes_to_draw:
            norm_factor = 1.0 / raw_data[lane.name].max()
            raw_data[lane.name] *= norm_factor
        return raw_data


class NormLaneMaxLog2(_Normalization):
    """Normalize to the maximum value of the regions in each lane, then log2"""

    def __init__(self):
        self.name = "NormLaneMaxLog2"

    def get_dependencies(self, lanes_to_draw):
        return []

    def calc(self, lanes_to_draw, raw_data):
        for lane in lanes_to_draw:
            norm_factor = 1.0 / raw_data[lane.name].max()
            raw_data[lane.name] *= norm_factor
            raw_data[lane.name] = np.log2(raw_data[lane.name] + 1)
        return raw_data


class NormPerPeak(_Normalization):
    """Highest value in each peak is 1, lowest is 0"""

    name = "Norm_PerPeak"

    def get_dependencies(self, lanes_to_draw):
        return [x.count_aligned_reads() for x in lanes_to_draw]

    def calc(self, lanes_to_draw, raw_data):
        for lane in lanes_to_draw:
            data = raw_data[lane.name]
            minimum = data.min(axis=1)
            maximum = data.max(axis=1)
            data = data.transpose()
            data = data - minimum  # start from 0
            data = data / (maximum - minimum)  # norm to 0..1
            data = data.transpose()
            raw_data[lane.name] = data
        return raw_data


class NormPerRow(_Normalization):
    """Highest value in each row (ie in each peak across samples is 1, lowest is 0"""

    name = "NormPerRow"

    def get_dependencies(self, lanes_to_draw):
        return []

    def calc(self, lanes_to_draw, raw_data):
        maxima = {}
        minima = {}
        for lane in lanes_to_draw:
            maxima[lane.name] = raw_data[lane.name].max(axis=1)
            minima[lane.name] = raw_data[lane.name].min(axis=1)
        maxima = np.array(pd.DataFrame(maxima).max(axis=1))
        minima = np.array(pd.DataFrame(minima).max(axis=1))

        for lane in lanes_to_draw:
            data = raw_data[lane.name]
            data = data.transpose()
            data = data - minima  # start from 0
            data = data / (maxima - minima)  # norm to 0..1
            data = data.transpose()
            raw_data[lane.name] = data
        return raw_data


class NormPerRowTPM(_Normalization):
    """Highest value in each row (ie in each peak across samples is 1, lowest is 0),
    lanes are first converted to TPMs based on lane.get_aligned_read_count()"""

    name = "NormPerRowTPM"

    def get_dependencies(self, lanes_to_draw):
        return [x.count_aligned_reads() for x in lanes_to_draw]

    def calc(self, lanes_to_draw, raw_data):
        _apply_tpm(lanes_to_draw, raw_data)
        maxima = {}
        minima = {}
        for lane in lanes_to_draw:
            maxima[lane.name] = raw_data[lane.name].max(axis=1)
            minima[lane.name] = raw_data[lane.name].min(axis=1)
        maxima = np.array(pd.DataFrame(maxima).max(axis=1))
        minima = np.array(pd.DataFrame(minima).max(axis=1))

        for lane in lanes_to_draw:
            data = raw_data[lane.name]
            data = data.transpose()
            data = data - minima  # start from 0
            data = data / (maxima - minima)  # norm to 0..1
            data = data.transpose()
            raw_data[lane.name] = data
        return raw_data


class NormLaneQuantile(_Normalization):
    """Normalize so that everything above the quantile is max
    Start high, with 0.99 for example, when trying different values
    """

    def __init__(self, quantile):
        self.quantile = quantile
        self.name = "NormLaneQuantile_%s" % quantile

    def calc(self, lanes_to_draw, raw_data):
        _apply_tpm(lanes_to_draw, raw_data)
        for lane in lanes_to_draw:
            data = raw_data[lane.name]
            q = np.percentile(data, self.quantile * 100)
            data[data > q] = q
            raw_data[lane.name] = data
        return raw_data

    def get_dependencies(self, lanes_to_draw):
        return [x.count_aligned_reads() for x in lanes_to_draw]


class NormLaneQuantileIthLane(_Normalization):
    """Normalize TPM so that everything above the quantile is max
    But only use the quantile from the Ith Lane
    Start high, with 0.99 for example, when trying different values
    """

    def __init__(self, quantile, ith_lane):
        self.quantile = quantile
        self.name = "NormLaneQuantileIthLane_%i_%s" % (ith_lane, quantile)
        self.ith = ith_lane

    def calc(self, lanes_to_draw, raw_data):
        _apply_tpm(lanes_to_draw, raw_data)
        import pickle

        with open("debug.dat", "wb") as op:
            pickle.dump(raw_data[lanes_to_draw[self.ith].name], op)
        q = np.percentile(raw_data[lanes_to_draw[self.ith].name], self.quantile * 100)
        for lane in lanes_to_draw:
            data = raw_data[lane.name]
            data[data > q] = q
            raw_data[lane.name] = data
        return raw_data

    def get_dependencies(self, lanes_to_draw):
        return [x.count_aligned_reads() for x in lanes_to_draw]
