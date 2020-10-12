import pandas as pd
import numpy as np


class RegionAsIs:
    """Take the regions as they are.
        Explodes if they are not all the same size!
        """

    def __init__(self):
        self.name = "RegionAsIs"

    def calc(self, gr):
        """Must return a pandas dataframe with chr, start, stop, flip"""
        starts = gr.df["start"].astype(int)
        stops = gr.df["stop"].astype(int)
        chrs = gr.df["chr"]
        if len((stops - starts).unique()) > 1:
            raise ValueError(
                "Not all input regions were the same size - can't use RegionAsIS"
            )

        return pd.DataFrame(
            {"chr": chrs, "start": starts, "stop": stops, "flip": False}
        )

    def get_dependencies(self, gr):
        return [gr.load()]


class RegionFromCenter:
    """Take the regions as they are, cookie cut into center +- xbp
    No region get's flipped.
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
        return pd.DataFrame({"chr": chrs, "start": left, "stop": right, "flip": False})

    def get_dependencies(self, gr):
        return [gr.load()]


class RegionFromSummit:
    """Take the regions from their summits as defined by @summit_annotator (mbf_genomics.regions.annotators.Summit*,
    basically an annotator defining an offset to start),
    then +- 0.5 * total_size
    No region get's flipped.
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
        return pd.DataFrame({"chr": chrs, "start": left, "stop": right, "flip": False})

    def get_dependencies(self, gr):
        return [gr.add_annotator(self.summit_annotator), gr.load()]


class RegionFromCenterFlipByNextGene:
    """Todo"""

    def __init__(self, total_size):
        raise NotImplementedError()


class RegionSample(object):
    """Subsample regions (without replacement).
    (ie. turn this into fewer regions)

    Uses a a nested RegionStrategy for initial conversion,
    then keeps only randomly choosen ones

    """

    def __init__(self, inner_region_strategy, ratio_or_count, seed=500):
        """
        @inner_region_strategy: another region strategy to preprocess the regions first

        @ratio_or_count either a number 0.0 .. 1.0, in which case we subsample
        to total count * @ratio_or_count, or a hard number to sample to

        @seed: initial seed for random number generator - to make heatmaps repeatable
        """
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

    # todo: handle changing seed by rebuilding the heatmap.
