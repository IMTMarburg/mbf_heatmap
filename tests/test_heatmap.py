import pytest
import pypipegraph as ppg
import numpy as np
import pandas as pd
from mbf_genomics import DelayedDataFrame
from mbf_qualitycontrol.testing import assert_image_equal
from mbf_heatmap import HeatmapPlot, heatmap_order, heatmap_norm

from pypipegraph.testing import run_pipegraph


@pytest.mark.usefixtures("both_ppg_and_no_ppg_no_qc")
class TestComplete:
    def test_very_simple(self):
        df = pd.DataFrame(
            {
                "a1": [0, 1, 2],
                "a2": [0.5, 1.5, 2.5],
                "b1": [2, 1, 0],
                "b2": [2.5, 0.5, 1],
            }
        )
        ddf = DelayedDataFrame("test", df)
        of = "test.png"
        h = HeatmapPlot(
            ddf, df.columns, of, heatmap_norm.Unchanged(), heatmap_order.Unchanged()
        )
        run_pipegraph()
        assert_image_equal(h.output_filename)

    def test_hierarchical_pearson(self):
        df = pd.DataFrame(
            {
                "a1": [0, 1, 2],
                "a2": [0.5, 1.5, 2.5],
                "b1": [2, 1, 0],
                "b2": [0.5, 0.5, 1],
            }
        )
        df = df.sample(200, replace=True, random_state=500)
        np.random.seed(500)
        df += np.random.normal(0, 1, df.shape)
        ddf = DelayedDataFrame("test", df)
        of = "test.png"
        h = HeatmapPlot(
            ddf, df.columns, of, heatmap_norm.Unchanged(), heatmap_order.HierarchicalPearson ()
        )
        run_pipegraph()
        assert_image_equal(h.output_filename)

