import pypipegraph as ppg
import pytest
import pandas as pd
import numpy as np
import mbf_heatmap
import mbf_genomics
import mbf_sampledata
import mbf_align
from mbf_heatmap.chipseq import regions, smooth, norm, order
from mbf_qualitycontrol.testing import assert_image_equal
from mbf_genomics.testing import MockGenome


def get_human_22_fake_genome():
    import gzip

    genes = pd.read_msgpack(
        gzip.GzipFile(
            mbf_sampledata.get_sample_path("mbf_align/hs_22_genes.msgpack.gz")
        )
    ).reset_index()
    tr = pd.read_msgpack(
        gzip.GzipFile(
            mbf_sampledata.get_sample_path("mbf_align/hs_22_transcripts.msgpack.gz")
        )
    ).reset_index()
    genes["chr"] = "chr22"
    tr["chr"] = "chr22"
    return MockGenome(
        df_genes=genes, df_transcripts=tr, chr_lengths={"chr22": 50_818_468}
    )


class TestHeatmapChipSeq:
    def test_simple(self, new_pipegraph_no_qc):
        genome = get_human_22_fake_genome()
        start = 17750239
        df = pd.DataFrame(
            [
                {"chr": "chr22", "start": start, "stop": start + 1000},
                {"chr": "chr22", "start": start + 20000, "stop": start + 20000 + 1000},
                {"chr": "chr22", "start": start + 30000, "stop": start + 30000 + 1000},
            ]
        )
        plot_regions = mbf_genomics.regions.GenomicRegions(
            "testregions", lambda: df, [], genome
        )
        lane1 = mbf_align.lanes.AlignedSample(
            "one",
            mbf_sampledata.get_sample_path("mbf_align/chipseq_chr22.bam"),
            genome,
            False,
            None,
        )
        lane2 = mbf_align.lanes.AlignedSample(
            "two",
            mbf_sampledata.get_sample_path("mbf_align/chipseq_chr22.bam"),
            genome,
            False,
            None,
        )

        h = mbf_heatmap.chipseq.Heatmap(
            plot_regions,
            [lane1, lane2],
            region_strategy=regions.RegionAsIs(),
            smoothing_strategy=smooth.SmoothRaw(),
        )
        fn = "test.png"
        h.plot(fn, norm.AsIs(), order.AsIs())
        ppg.run_pipegraph()
        assert_image_equal(fn)

    def test_smooth(self, new_pipegraph_no_qc):
        genome = get_human_22_fake_genome()
        df = pd.DataFrame(
            [
                {
                    "chr": "chr22",
                    "start": 36925 * 1000 - 1000,
                    "stop": 36925 * 1000 + 1000,
                },
                {
                    "chr": "chr22",
                    "start": 31485 * 1000 - 2000,
                    "stop": 31485 * 1000 + 2000,
                },
                {"chr": "chr22", "start": 41842 * 1000, "stop": (41842 * 1000) + 1},
            ]
        )
        plot_regions = mbf_genomics.regions.GenomicRegions(
            "testregions", lambda: df, [], genome
        )
        lane1 = mbf_align.lanes.AlignedSample(
            "one",
            mbf_sampledata.get_sample_path("mbf_align/chipseq_chr22.bam"),
            genome,
            False,
            None,
        )
        lane2 = mbf_align.lanes.AlignedSample(
            "two",
            mbf_sampledata.get_sample_path("mbf_align/chipseq_chr22.bam"),
            genome,
            False,
            None,
        )

        h = mbf_heatmap.chipseq.Heatmap(
            plot_regions,
            [lane1, lane2],
            region_strategy=regions.RegionFromCenter(1000),
            smoothing_strategy=smooth.SmoothExtendedReads(),
        )
        fn = "test.png"
        h.plot(fn, norm.AsIs(), order.FirstLaneSum())
        ppg.run_pipegraph()
        assert_image_equal(fn)


class TestSmooth:
    def test_raw(self, new_pipegraph):
        genome = get_human_22_fake_genome()
        lane1 = mbf_align.lanes.AlignedSample(
            "one",
            mbf_sampledata.get_sample_path("mbf_align/chipseq_chr22.bam"),
            genome,
            False,
            None,
        )
        start = 41842000
        regions = pd.DataFrame(
            {
                "chr": ["chr22"],
                "start": [
                    start,
                ],
                "stop": [start + 1000],
            }
        )
        calculated = smooth.SmoothRaw().calc(regions, lane1)
        should = np.zeros(1000)
        known = [
            (41842170, True, [(0, 36)]),
            (41842241, False, [(0, 36)]),
            (41842399, False, [(0, 36)]),
            (41842416, False, [(0, 36)]),
            (41842602, True, [(0, 36)]),
            (41842687, False, [(0, 36)]),
            (41842689, True, [(0, 36)]),
            (41842730, True, [(0, 36)]),
            (41842750, False, [(0, 36)]),
            (41842770, True, [(0, 36)]),
            (41842796, True, [(0, 36)]),
            (41842942, False, [(0, 36)]),
            (41842985, False, [(0, 36)]),
        ]

        for pos, is_reverse, cigar in known:
            pos -= start
            # orientation does not matter for non-extended reads
            should[pos : pos + cigar[0][1]] += 1
        should = should.reshape((1, 1000))
        assert should.shape == calculated.shape
        if (should != calculated).any():
            for ii in range(1000):
                if should[0, ii] != calculated[0, ii]:
                    print(ii, should[0, ii], calculated[0, ii])
        assert (should == calculated).all()

    def test_extended(self, new_pipegraph):
        genome = get_human_22_fake_genome()
        lane1 = mbf_align.lanes.AlignedSample(
            "one",
            mbf_sampledata.get_sample_path("mbf_align/chipseq_chr22.bam"),
            genome,
            False,
            None,
        )
        start = 41842000
        regions = pd.DataFrame(
            {
                "chr": ["chr22"],
                "start": [
                    start,
                ],
                "stop": [start + 1000],
            }
        )
        extend = 10
        calculated = smooth.SmoothExtendedReads(extend).calc(regions, lane1)
        should = np.zeros(1000)
        known = [
            (41842170, True, [(0, 36)]),
            (41842241, False, [(0, 36)]),
            (41842399, False, [(0, 36)]),
            (41842416, False, [(0, 36)]),
            (41842602, True, [(0, 36)]),
            (41842687, False, [(0, 36)]),
            (41842689, True, [(0, 36)]),
            (41842730, True, [(0, 36)]),
            (41842750, False, [(0, 36)]),
            (41842770, True, [(0, 36)]),
            (41842796, True, [(0, 36)]),
            (41842942, False, [(0, 36)]),
            (41842985, False, [(0, 36)]),
        ]

        for pos, is_reverse, cigar in known:
            pos -= start
            print(pos)
            if is_reverse:  # downstream verlaengern!
                should[pos - extend : pos + cigar[0][1]] += 1
            else:
                should[pos : pos + cigar[0][1] + extend] += 1
        should = should.reshape((1, 1000))
        assert should.shape == calculated.shape
        if (should != calculated).any():
            for ii in range(1000):
                if should[0, ii] != calculated[0, ii]:
                    print(ii, should[0, ii], calculated[0, ii])
        assert (should == calculated).all()

    def test_extended_larger_then_region(self, new_pipegraph):
        genome = get_human_22_fake_genome()
        lane1 = mbf_align.lanes.AlignedSample(
            "one",
            mbf_sampledata.get_sample_path("mbf_align/chipseq_chr22.bam"),
            genome,
            False,
            None,
        )
        start = 41842000
        regions = pd.DataFrame(
            {
                "chr": ["chr22"],
                "start": [
                    start,
                ],
                "stop": [start + 1000],
            }
        )
        extend = 1500
        calculated = smooth.SmoothExtendedReads(extend).calc(regions, lane1)
        should = np.zeros(1000)
        known = [
            (41842170, True, [(0, 36)]),
            (41842241, False, [(0, 36)]),
            (41842399, False, [(0, 36)]),
            (41842416, False, [(0, 36)]),
            (41842602, True, [(0, 36)]),
            (41842687, False, [(0, 36)]),
            (41842689, True, [(0, 36)]),
            (41842730, True, [(0, 36)]),
            (41842750, False, [(0, 36)]),
            (41842770, True, [(0, 36)]),
            (41842796, True, [(0, 36)]),
            (41842942, False, [(0, 36)]),
            (41842985, False, [(0, 36)]),
        ]

        for pos, is_reverse, cigar in known:
            pos -= start
            if is_reverse:
                should[max(0, pos - extend) : min(1000, pos + cigar[0][1])] += 1
            else:
                should[max(pos, 0) : min(1000, pos + cigar[0][1] + extend)] += 1
        should = should.reshape((1, 1000))
        assert should.shape == calculated.shape
        if (should != calculated).any():
            for ii in range(1000):
                if should[0, ii] != calculated[0, ii]:
                    print(ii, should[0, ii], calculated[0, ii])
        assert (should == calculated).all()

    def test_extended_minus_background(self, new_pipegraph):
        genome = get_human_22_fake_genome()
        lane1 = mbf_align.lanes.AlignedSample(
            "one",
            mbf_sampledata.get_sample_path("mbf_align/chipseq_chr22.bam"),
            genome,
            False,
            None,
        )
        start = 41842000
        regions = pd.DataFrame(
            {
                "chr": ["chr22"],
                "start": [
                    start,
                ],
                "stop": [start + 1000],
            }
        )
        extend = 10
        sermb = smooth.SmoothExtendedReadsMinusBackground({lane1.name: lane1}, extend)
        calculated = sermb.calc(regions, lane1)
        should = np.zeros((1, 1000))
        assert (should == calculated).all()
        assert lane1.load() in sermb.get_dependencies(lane1)


class TestOrder:
    def test_ithlane_sum(self, new_pipegraph):
        genome = get_human_22_fake_genome()
        start = 17750239
        df = pd.DataFrame(
            [
                {"chr": "chr22", "start": start, "stop": start + 1000},
                {"chr": "chr22", "start": start + 20000, "stop": start + 20000 + 1000},
                {"chr": "chr22", "start": start + 30000, "stop": start + 30000 + 1000},
            ]
        )
        lane1 = mbf_align.lanes.AlignedSample(
            "one",
            mbf_sampledata.get_sample_path("mbf_align/chipseq_chr22.bam"),
            genome,
            False,
            None,
        )
        lane2 = mbf_align.lanes.AlignedSample(
            "two",
            mbf_sampledata.get_sample_path("mbf_align/chipseq_chr22.bam"),
            genome,
            False,
            None,
        )
        with pytest.raises(AttributeError):
            order.IthLaneSum(lane1.name)

        o = order.IthLaneSum(1)
        # raw_data = {lane1.name: smooth.SmoothRaw().calc(df, lane1)}
        raw_data = {
            lane1.name: np.array(
                [
                    [0, 0, 4, 0],
                    [1, 1, 1, 0],
                    [1, 0, 0, 0],
                ]
            )
        }

        print(raw_data)
        print(raw_data[lane1.name].sum(axis=1))
        lanes = {lane1.name: lane1}
        lanes[lane2.name] = lane2  # make sure they have a defined order!
        norm_data = norm.AsIs().calc(lanes, raw_data)
        plot_regions = mbf_genomics.regions.GenomicRegions(
            "testregions", lambda: df, [], genome
        )

        with pytest.raises(KeyError):
            o.calc(
                plot_regions,
                {lane1.name: lane1, lane2.name: lane2},
                raw_data,
                norm_data,
            )

        o = order.IthLaneSum(lane2)
        with pytest.raises(KeyError):
            o.calc(plot_regions, {lane1.name: lane1}, raw_data, norm_data)

        raw_data[lane2.name] = raw_data[lane1.name].copy()
        res_order, clusters = o.calc(plot_regions, lanes, raw_data, norm_data)
        assert clusters is None
        assert (
            res_order == [2, 1, 0]
        ).all()  # remember, from top to bottom in plotting later on.

        raw_data[lane2.name] = np.array(
            [
                [0, 0, 0, 0],
                [4, 1, 1, 0],
                [1, 0, 0, 0],
            ]
        )
        o = order.IthLaneSum(0)
        res_order, clusters = o.calc(plot_regions, lanes, raw_data, norm_data)
        assert (
            res_order == [2, 1, 0]
        ).all()  # remember, from top to bottom in plotting later on.

        o = order.IthLaneSum(1)
        res_order, clusters = o.calc(plot_regions, lanes, raw_data, norm_data)

        assert (
            res_order == [0, 2, 1]
        ).all()  # remember, from top to bottom in plotting later on.

    def test_ithlane_max(self, new_pipegraph):
        genome = get_human_22_fake_genome()
        start = 17750239
        df = pd.DataFrame(
            [
                {"chr": "chr22", "start": start, "stop": start + 1000},
                {"chr": "chr22", "start": start + 20000, "stop": start + 20000 + 1000},
                {"chr": "chr22", "start": start + 30000, "stop": start + 30000 + 1000},
            ]
        )
        lane1 = mbf_align.lanes.AlignedSample(
            "one",
            mbf_sampledata.get_sample_path("mbf_align/chipseq_chr22.bam"),
            genome,
            False,
            None,
        )
        lane2 = mbf_align.lanes.AlignedSample(
            "two",
            mbf_sampledata.get_sample_path("mbf_align/chipseq_chr22.bam"),
            genome,
            False,
            None,
        )
        with pytest.raises(AttributeError):
            order.IthLaneMax(lane1.name)

        o = order.IthLaneMax(1)
        # raw_data = {lane1.name: smooth.SmoothRaw().calc(df, lane1)}
        raw_data = {
            lane1.name: np.array(
                [
                    [0, 0, 5, 0],
                    [2, 1, 1, 1],
                    [1, 0, 0, 0],
                ]
            )
        }

        print(raw_data)
        print(raw_data[lane1.name].max(axis=1))
        lanes = {lane1.name: lane1}
        lanes[lane2.name] = lane2
        norm_data = norm.AsIs().calc(lanes, raw_data)
        plot_regions = mbf_genomics.regions.GenomicRegions(
            "testregions", lambda: df, [], genome
        )

        with pytest.raises(KeyError):
            o.calc(
                plot_regions,
                {lane1.name: lane1, lane2.name: lane2},
                raw_data,
                norm_data,
            )

        o = order.IthLaneMax(lane2)
        with pytest.raises(KeyError):
            o.calc(plot_regions, {lane1.name: lane1}, raw_data, norm_data)

        raw_data[lane2.name] = raw_data[lane1.name].copy()
        res_order, clusters = o.calc(plot_regions, lanes, raw_data, norm_data)
        assert clusters is None
        assert (
            res_order == [2, 1, 0]
        ).all()  # remember, from top to bottom in plotting later on.

        raw_data[lane2.name] = np.array(
            [
                [0, 0, 0, 0],
                [5, 1, 1, 0],
                [1, 0, 0, 4],
            ]
        )
        o = order.IthLaneMax(0)
        res_order, clusters = o.calc(plot_regions, lanes, raw_data, norm_data)
        assert (
            res_order == [2, 1, 0]
        ).all()  # remember, from top to bottom in plotting later on.

        o = order.IthLaneMax(1)
        res_order, clusters = o.calc(plot_regions, lanes, raw_data, norm_data)

        assert (
            res_order == [0, 2, 1]
        ).all()  # remember, from top to bottom in plotting later on.

    def test_by_column(self, new_pipegraph_no_qc):
        genome = get_human_22_fake_genome()
        start = 17750239
        df = pd.DataFrame(
            [
                {
                    "chr": "chr22",
                    "start": start,
                    "stop": start + 1000,
                    "colA": "a",
                },
                {
                    "chr": "chr22",
                    "start": start + 20000,
                    "stop": start + 20000 + 1000,
                    "colA": "c",
                },
                {
                    "chr": "chr22",
                    "start": start + 30000,
                    "stop": start + 30000 + 1000,
                    "colA": "b",
                },
            ]
        )
        lane1 = mbf_align.lanes.AlignedSample(
            "one",
            mbf_sampledata.get_sample_path("mbf_align/chipseq_chr22.bam"),
            genome,
            False,
            None,
        )
        lanes = {lane1.name: lane1}
        o = order.ByAnnotator("colA", func=lambda x: [ord(y) for y in x])
        raw_data = {
            lane1.name: np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            )
        }
        plot_regions = mbf_genomics.regions.GenomicRegions(
            "testregions", lambda: df, [], genome
        )
        ppg.JobGeneratingJob("shu", lambda: None).depends_on(plot_regions.load())
        ppg.run_pipegraph()
        plot_regions._load()

        norm_data = norm.AsIs().calc(lanes, raw_data)
        res_order, clusters = o.calc(plot_regions, lanes, raw_data, norm_data)
        assert (res_order == [0, 2, 1]).all()

    def test_by_annotator(self, new_pipegraph_no_qc):
        genome = get_human_22_fake_genome()
        start = 17750239
        df = pd.DataFrame(
            [
                {
                    "chr": "chr22",
                    "start": start,
                    "stop": start + 1000,
                },
                {
                    "chr": "chr22",
                    "start": start + 20000,
                    "stop": start + 20000 + 1000,
                },
                {
                    "chr": "chr22",
                    "start": start + 30000,
                    "stop": start + 30000 + 1000,
                },
            ]
        )
        lane1 = mbf_align.lanes.AlignedSample(
            "one",
            mbf_sampledata.get_sample_path("mbf_align/chipseq_chr22.bam"),
            genome,
            False,
            None,
        )
        lanes = {lane1.name: lane1}
        raw_data = {
            lane1.name: np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            )
        }
        plot_regions = mbf_genomics.regions.GenomicRegions(
            "testregions", lambda: df, [], genome
        )

        class FakeAnno(mbf_genomics.annotator.Annotator):
            columns = ["colA"]

            def calc(self, df):
                return pd.Series([1, 3, 2])

        o = order.ByAnnotator(FakeAnno())
        ppg.JobGeneratingJob("shu", lambda: None).depends_on(
            o.get_dependencies(plot_regions, lanes)[0]
        )
        ppg.run_pipegraph()
        plot_regions._load()

        norm_data = norm.AsIs().calc(lanes, raw_data)
        res_order, clusters = o.calc(plot_regions, lanes, raw_data, norm_data)
        assert (res_order == [0, 2, 1]).all()
