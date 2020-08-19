# flake8: noqa
if False:  # noqa
    import pandas as pd
    import pypipegraph as ppg
    from .plots import get_coverage_vector

    import numpy as np

    def _apply_tpm(lanes_to_draw, raw_data):
        """Convert read counts in raw_data into TPMs - in situ"""
        for lane in lanes_to_draw:
            norm_factor = 1e6 / lane.get_aligned_read_count()
            raw_data[lane.name] = raw_data[lane.name] * norm_factor

    class SmoothExtendedReads(object):
        """Each read extended by x bp in 3' direction"""

        def __init__(self, extend_by_bp=200):
            self.name = "Smooth_Extended_%ibp" % extend_by_bp
            self.extend_by_bp = extend_by_bp

        def get_dependencies(self, lane):
            deps = [lane.align()]
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
                    lane, row["chr"], row["start"], row["stop"], self.extend_by_bp
                )
                if len(signal) != row["stop"] - row["start"]:
                    raise ValueError(
                        "Signal had wrong length:\nrow: %s,\nsignal_shape: %s,\nstop-start=%s"
                        % (row, signal.shape, row["stop"] - row["start"])
                    )
                result.append(signal)
            return np.array(result)

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
            deps = [lane.align()]
            deps.append(self.background[lane.name].align())
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
                    lane, row["chr"], row["start"], row["stop"], self.extend_by_bp
                )
                background_signal = get_coverage_vector(
                    bg_lane, row["chr"], row["start"], row["stop"], self.extend_by_bp
                )
                result.append(signal - background_signal)
            return np.array(result)

    class SmoothRaw(SmoothExtendedReads):
        """just the reads, no smoothing"""

        def __init__(self):
            self.name = "Smooth_Raw"
            self.extend_by_bp = 0

    class NormLaneTPM:
        """Normalize to TPM based on lane.get_aligned_read_count"""

        name = "Norm_Lane_TPM"

        def get_dependencies(self, lanes_to_draw):
            return [x.count_aligned_reads() for x in lanes_to_draw]

        def calc(self, lanes_to_draw, raw_data):
            _apply_tpm(lanes_to_draw, raw_data)
            return raw_data

    class NormLaneTPMInterpolate:
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

    class NormLaneMax:
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

    class NormLaneMaxLog2:
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

    class NormPerPeak:
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

    class NormPerRow:
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

    class NormPerRowTPM:
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

    class NormLaneQuantile:
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

    class NormLaneQuantileIthLane:
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
            q = np.percentile(
                raw_data[lanes_to_draw[self.ith].name], self.quantile * 100
            )
            for lane in lanes_to_draw:
                data = raw_data[lane.name]
                data[data > q] = q
                raw_data[lane.name] = data
            return raw_data

        def get_dependencies(self, lanes_to_draw):
            return [x.count_aligned_reads() for x in lanes_to_draw]

    class RegionFromCenter:
        """Take the regions as they are, cookie cut into center +- x @bp
        No region get's flipped
        """

        def __init__(self, total_size):
            self.total_size = total_size
            self.name = "Region_From_Center_%i" % self.total_size

        def calc(self, gr):
            """Must return a pandas dataframe with chr, start, stop, flip"""
            starts = gr.df["start"]
            stops = gr.df["stop"]
            chrs = gr.df["chr"]
            centers = ((stops - starts) / 2.0 + starts).astype(int)
            left = centers - self.total_size / 2
            right = centers + self.total_size / 2
            return pd.DataFrame(
                {"chr": chrs, "start": left, "stop": right, "flip": False}
            )

        def get_dependencies(self, gr):
            return [gr.load()]

    class RegionFromSummit:
        """Take the regions from their summits as defined by @summit_annotator,
        then +- 0.5 * total_size
        No region get's flipped
        """

        def __init__(self, total_size, summit_annotator):
            self.name = "RegionFromSummit_%i_%s" % (
                total_size,
                summit_annotator.column_names[0],
            )
            self.total_size = total_size
            self.summit_annotator = summit_annotator

        def calc(self, gr):
            """Must return a pandas dataframe with chr, start, stop, flip"""
            starts = gr.df["start"]
            summit = gr.df[self.summit_annotator.column_name]
            chrs = gr.df["chr"]
            centers = starts + summit
            left = centers - self.total_size / 2
            right = centers + self.total_size / 2
            if len(set(right - left)) > 1:
                raise ValueError("not all regions were created with the same size")
            return pd.DataFrame(
                {"chr": chrs, "start": left, "stop": right, "flip": False}
            )

        def get_dependencies(self, gr):
            return [gr.add_annotator(self.summit_annotator), gr.load()]

    class RegionFromCenterFlipByNextGene:
        def __init__(self, total_size):
            raise NotImplementedError()

    class RegionSample(object):
        """Subsample regions (without replacement). Uses an inner RegionStrategy then keeps only randomly choosen ones"""

        def __init__(self, inner_region_strategy, ratio_or_count, seed=500):
            self.name = "Region_Sample_%f_%i_%s" % (
                ratio_or_count,
                seed,
                inner_region_strategy.name,
            )
            self.inner_region_strategy = inner_region_strategy
            self.ratio_or_count = ratio_or_count
            self.seed = seed

        def calc(self, gr):
            res = self.inner_region_strategy.calc(gr)
            np.random.seed(self.seed)
            if self.ratio_or_count < 1:
                count = int(len(res) * self.ratio_or_count)
            else:
                count = self.ratio_or_count
            return res.sample(n=count)

        def get_dependencies(self, gr):
            return self.inner_region_strategy.get_dependencies(gr)

    class Heatmap:
        def __init__(
            self,
            gr_to_draw,
            lanes_to_draw,
            region_strategy=RegionFromCenter(2000),
            smoothing_strategy=SmoothExtendedReads(200),
        ):
            """A one-line-per-region chipseq signal-intensity via color heatmap
            object.

            Parameters:
            @gr_to_draw:
                Which genomic regions do you want to draw? Each entry must be the same size!
            @lanes_to_draw:
                Which AlignedLanes shall we draw, left to right?
            @region_strategy:
                How to convert the regions intervals into the same-sized regions to plot (One of the Region_* classes)
            @smoothing_strategy:
                How shall the reads be proprocessed (e.g. extended, background substracted...) - one of the Smooth_* classes from this file
            """
            self.gr_to_draw = gr_to_draw
            self.lanes_to_draw = lanes_to_draw
            if len(set([x.name for x in lanes_to_draw])) != len(lanes_to_draw):
                raise ValueError("Duplicate names")
            self.region_strategy = region_strategy
            self.smoothing_strategy = smoothing_strategy

        def plot(
            self,
            output_filename,
            normalization_strategy=NormLaneTPM(),
            order_strategy=OrderFirstLaneSignalSum(),
            plot_strategy=Plot_Matplotlib(),
            names=None,
            **plot_options
        ):
            """Plot the heatmap into @output_file
            Parameters:
                @output_file:
                    Where to plot the heatmap
                @normalization_strategy:
                    How shall the signal (from the smoothing_strategy) be normalized?
                @order_strategy:
                    In which order shall the regions be drawen (one of ther Order_* classes from this file)
                @plot_strategy:
                    Shall we use matplotlib or pyggplot to draw the heatmapy (Plot* classes)
                @names:
                    None - use aligned_lane.name
                    'short' - use aligned_lane.short_name
                    list - use names in order (see Heatmap.lanes_to_draw)
                    dictionary - partial lookup - either dict[lane], or lane.name if missing
                    function - called for each lane, with the lane being the sole parameter
                every other named paremeter get's passend to your Plot_* class - see them for details
                    """
            res = _HeatmapPlot(
                self,
                output_filename,
                normalization_strategy,
                order_strategy,
                plot_strategy,
                names,
                plot_options,
            )
            res()
            return res

        def calc_regions(self):
            def calc():
                return self.do_calc_regions()

            key = hashlib.md5(
                ",".join(
                    [self.gr_to_draw.name, self.region_strategy.name]
                    + list(set([x.name for x in self.lanes_to_draw]))
                ).encode()
            ).hexdigest()
            #  technically, we could share the regions job between heatmaps with the same regions but differen lanes
            # but we're using a CachedAttributeLoadingJob and that would.. .complicate things quite a bit
            common.ensure_path(os.path.join("cache", "ChipseqHeatmap", "regions"))
            of = os.path.join("cache", "ChipseqHeatmap", "regions", key)
            return ppg.CachedAttributeLoadingJob(of, self, "regions_", calc).depends_on(
                [
                    ppg.ParameterInvariant(
                        of, (self.region_strategy.name, self.gr_to_draw.name)
                    ),
                    ppg.FunctionInvariant(
                        "genomics.regions.heatmap."
                        + self.region_strategy.name
                        + "calc_func",
                        self.region_strategy.__class__.calc,
                    ),
                ]
                + self.region_strategy.get_dependencies(self.gr_to_draw)
            )

        def calc_raw_data(self):
            # we don't use a CachedAttributeLoadingJob so that we can compress the output.
            # don't knock that, it easily saves a gigabyte of data on a larger GR

            common.ensure_path(os.path.join("cache", "ChipseqHeatmap", "raw_data"))

            jobs = []
            smoothing_invariant = (
                ppg.FunctionInvariant(
                    "genomics.regions.heatmap."
                    + self.smoothing_strategy.name
                    + "calc_func",
                    self.smoothing_strategy.__class__.calc,
                ),
            )
            for lane in self.lanes_to_draw:
                key = ",".join(
                    [
                        self.gr_to_draw.name,
                        self.region_strategy.name,
                        self.smoothing_strategy.name,
                        lane.name,
                    ]
                )
                key = hashlib.md5(key.encode()).hexdigest()
                of = os.path.join("cache/", "ChipseqHeatmap", "raw_data", key + ".npz")

                def calc(lane=lane, of=of):
                    """Raw data is a dictionary: lane_name: 2d matrix"""
                    raw_data = {lane.name: self.do_calc_raw_data(lane)}
                    np.savez_compressed(of, **raw_data)

                jobs.append(
                    ppg.FileGeneratingJob(of, calc).depends_on(
                        [
                            ppg.ParameterInvariant(
                                of,
                                (
                                    self.smoothing_strategy.name,
                                    lane.name,
                                    self.gr_to_draw.name,
                                ),
                            ),
                            smoothing_invariant,
                            self.calc_regions(),
                            ppg.FunctionInvariant(
                                "genomics.regions.heatmap.do_calc_raw_data",
                                Heatmap.do_calc_raw_data,
                            ),
                        ]
                        + self.smoothing_strategy.get_dependencies(lane)
                    )
                )

            def load():
                result = {}
                for job in jobs:
                    npzfile = np.load(job.job_id)
                    for f in npzfile.files:
                        result[f] = npzfile[f]
                return result

            key = ",".join(
                [
                    self.gr_to_draw.name,
                    self.region_strategy.name,
                    self.smoothing_strategy.name,
                    ",".join(list(sorted([x.name for x in self.lanes_to_draw]))),
                ]
            )
            return ppg.AttributeLoadingJob(
                key + "_load", self, "raw_data_", load
            ).depends_on(jobs)

        def do_calc_regions(self):
            self.regions_ = self.region_strategy.calc(self.gr_to_draw)
            return self.regions_

        def do_calc_raw_data(self, lane):
            if not hasattr(self, "raw_data_"):
                self.raw_data_ = {}
            lane_raw_data = self.smoothing_strategy.calc(self.regions_, lane)
            self.raw_data_[lane.name] = lane_raw_data
            return lane_raw_data
