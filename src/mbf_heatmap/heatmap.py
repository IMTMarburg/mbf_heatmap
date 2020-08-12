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
    - How the regions are Ordered (OrderIthLaneSum, OrderClusterKMeans)


"""
import pypipegraph as ppg
from mbf_genomics.util import parse_a_or_c
from . import plot_strategies, heatmap_norm, heatmap_order
from mbf_genomics import DelayedDataFrame
from mbf_genomics.util import freeze


class HeatmapPlot:
    def __init__(
        self,
        ddf: DelayedDataFrame,
        columns,
        output_filename,
        normalization_strategy: heatmap_norm.NormStrategy,
        order_strategy: heatmap_order.OrderStrategy,
        names=None,
        plot_options={},
    ):
        """
        ddf: The genomir regions you want to plot with a minimum of chr, start, stop, columns
        columns: The columns from ddf to plot, in order.
        output_filename: a path to store the heatmap in
        normalization_strategy: the normalization strategy to apply,  see heatmap_norm
        order_strategy: how to order the rows from the ddf, see heatmap_order

        plot_options:

            show_cluster_ids - whether to show cluster ids as little colored dots at the left hand side
        """
        self.ddf = ddf
        self.columns = [parse_a_or_c(x) for x in columns]
        self.output_filename = ddf.pathify(output_filename)
        if not isinstance(normalization_strategy, heatmap_norm.NormStrategy):
            raise ValueError(
                "normalization_strategy must be a heatmap_norm.NormStrategy descendend"
            )
        self.normalization_strategy = normalization_strategy
        if not isinstance(order_strategy, heatmap_order.OrderStrategy):
            raise ValueError(
                "order_strategy must be a heatmap_norm.NormStrategy descendend"
            )

        self.order_strategy = order_strategy
        self.plot_strategy = plot_strategies.Plot_Matplotlib()
        self.names = names
        self.plot_options = plot_options
        self.plot()

    def plot(self):
        normed = self.normed_ddf(self.ddf)
        ordered = self.ordered_ddf(normed)
        names = self.handle_names()

        def plot():
            p = self.plot_strategy.plot(ordered.df, names, self.plot_options)
            self.plot_strategy.render(str(self.output_filename), p)

        if ppg.inside_ppg():
            ppg.util.global_pipegraph.quiet = False
            deps = [
                ordered.load(),
                ppg.FunctionInvariant(
                    "mbf_heatmap." + self.plot_strategy.name + "plot_func",
                    self.plot_strategy.__class__.plot,
                ),
                ppg.FunctionInvariant(
                    "mbf_heatmap" + self.plot_strategy.name + "render_func",
                    self.plot_strategy.__class__.render,
                ),
                ppg.ParameterInvariant(
                    self.output_filename, freeze((self.names, self.plot_options))
                ),
            ]
            return ppg.FileGeneratingJob(self.output_filename, plot).depends_on(deps)
        else:
            plot()
            return self.output_filename

    def normed_ddf(self, input_ddf):
        def load():
            df = input_ddf.df[[ac[1] for ac in self.columns]]
            normed_df = self.normalization_strategy.calc(
                df, [ac[1] for ac in self.columns]
            )
            return normed_df
        output_name = input_ddf.name + "_heatmap_" + self.normalization_strategy.name
        if ppg.inside_ppg():
            deps = [
                self.ddf.add_annotator(ac[0])
                for ac in self.columns
                if ac[0] is not None
            ] + [self.normalization_strategy.deps(), input_ddf.load(),
                 ppg.FunctionInvariant(output_name + '_calc', self.normalization_strategy.calc)]
        else:
            deps = []

        return DelayedDataFrame(
            output_name,
            load,
            deps,
            input_ddf.result_dir,
        )

    def ordered_ddf(self, input_ddf):
        def load():
            df = input_ddf.df[[ac[1] for ac in self.columns]]
            return self.order_strategy.calc(df, [ac[1] for ac in self.columns])
        output_name = input_ddf.name + self.order_strategy.name
        if ppg.inside_ppg():
            deps = [
                self.ddf.add_annotator(ac[0])
                for ac in self.columns
                if ac[0] is not None
            ] + [self.order_strategy.deps(), input_ddf.load(),
                 ppg.FunctionInvariant(output_name + '_calc', self.order_strategy.calc)]
        else:
            deps = []

        return DelayedDataFrame(
            output_name, load, deps, input_ddf.result_dir
        )

    def handle_names(self):
        if self.names is None:
            names = [
                getattr(ac[0], "plot_name", ac[1]) if ac[0] is not None else ac[1]
                for ac in self.columns
            ]
        elif isinstance(self.names, dict):
            if isinstance(iter(self.names.values()).next(), tuple):
                names = [self.names[ac] for ac in self.columns]
            else:  # byi column name
                names = [self.names[ac[1]] for ac in self.columns]
        elif isinstance(self.names, list):
            if len(self.names) != len(self.columns):
                raise ValueError("Name length did not match column length")
            names = self.names
        else:
            raise ValueError("Could not handle names %s" % (names,))
        return names
