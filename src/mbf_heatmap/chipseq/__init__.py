"""Support for heatmap images of chipseq data.

You need
    - a genomic regions which you want to plot (may be of differing size)
    - a number of AlignedLanes to plot.
You do
    - create a Heatmap object
    - call plot(output_filename,...) on it

You need to decide and pass in appropriate strategies:

    - How the regions get cookie cut (e.g. RegionFromCenter)
    - How the reads are counted (e.g. SmoothExtendedReads)
    - How the data is normalized (NormLaneTPMInterpolate is fast and sensible)
    - How the regions are Ordered (order.IthLaneSum, order.ClusterKMeans)
    """
# flake8: noqa
from pathlib import Path
import hashlib
import pandas as pd
from typing import List
import pypipegraph as ppg
import numpy as np
from . import regions
from . import smooth
from . import order
from . import norm
from .plot_strategies import Plot_Matplotlib


class Heatmap:
    def __init__(
        self,
        regions_to_draw,
        lanes_to_draw: List,
        region_strategy=regions.RegionFromCenter(2000),
        smoothing_strategy=smooth.SmoothExtendedReads(200),
    ):
        """
        A one-line-per-region, one column per (chipseq-) lane signal-intensity
        mapped to a color heatmap object.

        Parameters:
        @regions_to_draw:
            Which genomic regions do you want to draw? Will be processed by the @region_strategy
        @lanes_to_draw:
            Which Chipseq lanes shall we draw, left to right? a list
        @region_strategy:
            How to convert the regions intervals into the same-sized regions to plot (One of the Region_* classes)
        @smoothing_strategy:
            How shall the reads be proprocessed (e.g. extended, background substracted...) - one of the Smooth_*

        This is half the process of drawing a heatmap - just the parts that take the largest amout of time.
        """
        self.gr_to_draw = regions_to_draw
        self.lanes_to_draw = lanes_to_draw
        if len(set(lanes_to_draw)) != len(lanes_to_draw):
            raise ValueError("Duplicate lanes passed")
        self.region_strategy = region_strategy
        self.smoothing_strategy = smoothing_strategy
        self.cache_dir = self.cache_dir = Path("cache") / "ChipseqHeatmap"

    def plot(
        self,
        output_filename,
        normalization_strategy: norm._Normalization = norm.NormLaneTPM(),
        order_strategy: order._Order = order.FirstLaneSum(),
        names=None,
        plot_options: dict = None,
    ):
        """Plot the heatmap into @output_file
        Parameters:
            @output_file:
                Where to plot the heatmap (.png / .pdf)
            @normalization_strategy:
                How shall the signal (from the smoothing_strategy) be normalized? Use of the Norm_* classes
            @order_strategy:
                In which order shall the regions be drawen (one of ther Order_* classes)
            @names:
                None - use aligned_lane.name
                'short' - use aligned_lane.short_name
                list - use names in order (see Heatmap.lanes_to_draw)
                dictionary - partial lookup - either dict[lane], or lane.name if missing
                function - called get_name(lane) for each lane

            @plot_options: get's passed on to plot_strategies.Plot_Matplotlib, check there for valid parameters.

        """
        plot_strategy = Plot_Matplotlib()
        if plot_options is None:
            plot_options = {}
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
        of = self.cache_dir / "regions" / key
        of.parent.mkdir(exist_ok=True, parents=True)
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

        cache_dir = self.cache_dir / "raw_data"
        cache_dir.mkdir(exist_ok=True, parents=True)

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
            of = cache_dir / (key + ".npz")

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


class _HeatmapPlot:
    """This class encapsulates the heatmap parts that are specific to each rendering.
    The common parts such as raw data generation (and the user interface) are in Heatmap"""

    def __init__(
        self,
        heatmap,
        output_filename,
        normalization_strategy,
        order_strategy,
        plot_strategy,
        names,
        plot_options,
    ):
        """See Heatmap.plot for details"""
        self.heatmap = heatmap
        self.output_filename = Path(output_filename)
        self.output_filename.parent.mkdir(exist_ok=True, parents=True)
        self.name = "HeatmapPlot" + hashlib.md5(output_filename.encode()).hexdigest()
        ppg.util.assert_uniqueness_of_object(self)
        del self.name  # only used for uniqueness check...
        self.cache_dir = Path("cache") / "ChipseqHeatmap" / self.output_filename.name
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        self.normalization_strategy = normalization_strategy
        self.order_strategy = order_strategy
        self.plot_strategy = plot_strategy
        self.names = names
        self.plot_options = plot_options

    def __call__(self):
        norm_job = self.calc_norm_data()
        order_job = self.calc_order()
        names_in_order = [
            self.handle_name(self.names, x, ii)
            for (ii, x) in enumerate(self.heatmap.lanes_to_draw)
        ]

        def plot():
            p = self.do_plot()
            self.plot_strategy.render(self.output_filename, p)

        plot_job = ppg.FileGeneratingJob(self.output_filename, plot)
        plot_job.ignore_code_changes()
        plot_job.depends_on(norm_job)
        plot_job.depends_on(order_job)
        plot_job.depends_on(
            ppg.FunctionInvariant(
                "genomics.regions._HeatmapPlot.do_plot", _HeatmapPlot.do_plot
            )
        )
        plot_job.depends_on(
            ppg.FunctionInvariant(
                "genomics.regions.heatmap." + self.plot_strategy.name + "plot_func",
                self.plot_strategy.__class__.plot,
            )
        )
        plot_job.depends_on(
            ppg.FunctionInvariant(
                "genomics.regions.heatmap." + self.plot_strategy.name + "render_func",
                self.plot_strategy.__class__.render,
            )
        )
        plot_job.depends_on(self.heatmap.gr_to_draw.load())
        plot_job.depends_on(
            ppg.ParameterInvariant(
                self.output_filename,
                self.plot_strategy.get_parameters(
                    self.plot_options, self.heatmap.lanes_to_draw
                )
                + (names_in_order,),
            )
        )
        plot_job.depends_on(
            self.plot_strategy.get_dependencies(self, self.plot_options)
        )
        if hasattr(self.names, "__call__"):
            plot_job.depends_on(
                ppg.FunctionInvariant(self.output_filename + "_names", self.names)
            )

    def handle_name(self, names, aligned_lane, lane_no):
        if names is None:
            return aligned_lane.name
        elif names is "short":
            return aligned_lane.short_name
        elif isinstance(names, list):
            return names[lane_no]
        elif isinstance(names, dict):
            return names.get(aligned_lane, aligned_lane.name)
        elif hasattr(names, "__call__"):
            return names(aligned_lane)
        else:
            raise ValueError("Invalid parameter for names: %s" % (names,))

    def calc_norm_data(self):
        def calc():
            """Normalized data is a dictionary: lane_name: 2d matrix"""
            return self.do_calc_norm_data()

        of = self.cache_dir / "norm_data"
        return ppg.AttributeLoadingJob(of, self, "norm_data_", calc).depends_on(
            [
                ppg.ParameterInvariant(of, (self.normalization_strategy.name,)),
                self.heatmap.calc_raw_data(),
                ppg.FunctionInvariant(
                    "genomics.regions.heatmap."
                    + self.normalization_strategy.name
                    + "calc_func",
                    self.normalization_strategy.__class__.calc,
                ),
            ]
            + self.normalization_strategy.get_dependencies(self.heatmap.lanes_to_draw)
        )

    def calc_order(self):
        def calc():
            return self.do_calc_order()

        of = self.cache_dir / "order"
        deps = self.order_strategy.get_dependencies(
            self.heatmap.gr_to_draw, self.heatmap.lanes_to_draw
        )
        if len(deps) == 2:
            order_deps, order_params = deps
            order_func = None
        else:
            order_deps, order_params, order_func = deps

        return ppg.CachedAttributeLoadingJob(of, self, "order_", calc).depends_on(
            [
                self.heatmap.calc_raw_data(),
                self.calc_norm_data(),
                ppg.ParameterInvariant(of, (self.order_strategy.name,) + order_params),
                ppg.FunctionInvariant(
                    of.name + "_secondary_func", order_func
                ),
                ppg.FunctionInvariant(
                    "genomics.regions.heatmap."
                    + self.order_strategy.name
                    + "calc_func",
                    self.order_strategy.__class__.calc,
                ),
            ]
            + order_deps
        )

    def do_calc_norm_data(self):
        self.norm_data_ = self.normalization_strategy.calc(
            self.heatmap.lanes_to_draw, self.heatmap.raw_data_.copy()
        )
        return self.norm_data_

    def do_calc_order(self):
        self.order_ = self.order_strategy.calc(  # remember, the order is a tuple, 0 => index order to plot, 1 -> optional cluster number
            self.heatmap.gr_to_draw,
            self.heatmap.lanes_to_draw,
            self.heatmap.raw_data_.copy(),
            self.norm_data_,
        )
        if not isinstance(self.order_, tuple):
            raise ValueError(
                "Invalid self.order value, should have been a tuple, was: %s"
                % (self.order_,)
            )
        return self.order_

    def do_plot(self):
        """Return the prepared plot object, ready for rendering"""
        names_in_order = [
            self.handle_name(self.names, x, ii)
            for (ii, x) in enumerate(self.heatmap.lanes_to_draw)
        ]
        return self.plot_strategy.plot(
            self.heatmap.gr_to_draw,
            self.heatmap.lanes_to_draw,
            self.heatmap.raw_data_,
            self.norm_data_,
            self.order_,
            names_in_order,
            self.plot_options,
        )
