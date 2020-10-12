import pypipegraph as ppg
import numpy as np


def get_coverage_vector(bam, chr, start, stop, extend_reads_bp=0):
    # todo: replace with rust
    length = int(
        stop - start
    )  # here we still need floats for the calculation (if start and stop are floats, otherwise rounding down will make the array too short), but we need intgers for the np.zeros-array
    start = int(
        start
    )  # start and stop need to be integers, since we will not get reads from position x.5
    stop = int(stop)
    res = np.zeros(length, dtype=np.float)
    for read in bam.fetch(chr, max(start, 0), stop):
        if read.is_reverse:
            add_start = read.pos - extend_reads_bp
            add_stop = read.pos + read.qlen
        else:
            add_start = read.pos
            add_stop = read.pos + read.qlen + extend_reads_bp
        add_start = max(start, add_start)
        add_stop = min(stop, add_stop)
        res[add_start - start : add_stop - start] += 1
    return res


class SmoothExtendedReads(object):
    """Each read extended by x bp in 3' direction"""

    def __init__(self, extend_by_bp=200):
        self.name = "Smooth_Extended_%ibp" % extend_by_bp
        self.extend_by_bp = extend_by_bp

    def get_dependencies(self, lane):
        deps = [lane.load()]
        deps.append(
            ppg.FunctionInvariant(
                "genomics.regions.heatmaps." + self.name, self.__class__.calc
            )
        )
        return deps

    def calc(self, regions, lane):
        result = []
        for ii, row in regions.iterrows():
            signal = get_coverage_vector(
                lane.get_bam(), row["chr"], row["start"], row["stop"], self.extend_by_bp
            )
            if len(signal) != row["stop"] - row["start"]:  # pragma: no cover
                raise ValueError(  # pragma: no cover
                    "Signal had wrong length:\nrow: %s,\nsignal_shape: %s,\nstop-start=%s"
                    % (row, signal.shape, row["stop"] - row["start"])
                )
            result.append(signal)
        return np.vstack(result)


class SmoothExtendedReadsMinusBackground(object):
    """Each read extended by x bp in 3' direction"""

    def __init__(self, background_lanes, extend_by_bp=200):
        """
        @background_lanes a dictionary of lane_names to background lanes. lane_names means the foreground lanes as in lane.name!
        """
        self.name = "Smooth_Extended_%ibp_minus_background" % extend_by_bp
        self.extend_by_bp = extend_by_bp
        self.background = background_lanes

    def get_dependencies(self, lane):
        deps = [lane.load()]
        deps.append(self.background[lane.name].load())
        deps.append(
            ppg.FunctionInvariant(
                "genomics.regions.heatmaps." + self.name, self.__class__.calc
            )
        )
        return deps

    def calc(self, regions, lane):
        bg_lane = self.background[lane.name]
        result = []
        for ii, row in regions.iterrows():
            signal = get_coverage_vector(
                lane.get_bam(), row["chr"], row["start"], row["stop"], self.extend_by_bp
            )
            background_signal = get_coverage_vector(
                bg_lane.get_bam(),
                row["chr"],
                row["start"],
                row["stop"],
                self.extend_by_bp,
            )
            result.append(signal - background_signal)
        return np.array(result)


class SmoothRaw(SmoothExtendedReads):
    """just the reads, no smoothing"""

    def __init__(self):
        self.name = "Smooth_Raw"
        self.extend_by_bp = 0
