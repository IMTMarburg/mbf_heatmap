import pypipegraph as ppg
import numpy as np


class OrderStrategy:
    pass


class Unchanged(OrderStrategy):
    """straight pass through"""

    name = "Unchanged"

    def calc(self, df, columns):
        df = df[columns].assign(cluster=0)
        return df

    def deps(self):
        return []


class HierarchicalPearson(OrderStrategy):
    name = "HierarchicalPearson"

    def calc(self, df, columns):
        import scipy.cluster.hierarchy as hc

        matrix = df[columns].transpose().corr()
        z = hc.linkage(matrix)
        new_order = [x for x in hc.leaves_list(z)]
        df = df[columns].iloc[new_order].assign(cluster=0)
        return df

    def deps(self):
        return []


if False: # noqa
    from chipseq import _apply_tpm, pd, _HeatmapPlot

    class OrderAsIs:
        """Take the order as it was in gr_to_draw"""

        name = "Order_As_Is"

        def calc(self, gr, lanes, raw_data, norm_data):
            return (list(range(0, raw_data.values()[0].shape[0])), None)

        def get_dependencies(self, gr, lanes):
            return [gr.load()], (None,)

    def OrderFirstLaneSignalSum():
        return OrderIthLaneSum(0)

    class _OrderIthLane:
        def __init__(self, i, func, func_name):
            self.i = i
            self.func = func
            try:
                self.name = "OrderIthLane_%i_%s" % (i, func_name)
            except TypeError:  # raised on non integer...
                self.name = "OrderIthLane_%s_%s" % (i.name, func_name)

        def calc(self, gr, lanes, raw_data, norm_data):
            """ Returns the indices of the order, and an (optional) cluster number - None for no clustering
            """
            if hasattr(self.i, "name"):
                lane_name = self.i.name
            else:
                lane_name = lanes[self.i].name
            values = raw_data[lane_name]
            sums = self.func(values)
            return (np.argsort(sums), None)

        def get_dependencies(self, gr, lanes):
            return [gr.load()], (lanes[0].name,)

    class OrderByAnnotator:
        def __init__(self, annotator_to_order_by, ascending=True, func=None):
            self.annotator = annotator_to_order_by
            self.func = func
            self.ascending = ascending
            self.name = "OrderByAnnotator_%s" % (self.annotator.column_name)

        def calc(self, gr, lanes, raw_data, norm_data):
            """ Returns the indices of the order when sorted by annotator
            """
            df_sorted = gr.df
            if self.func is not None:
                df_sorted["sortme"] = self.func(
                    df_sorted[self.annotator.column_name].values
                )
                df_sorted = df_sorted.sort_values("sortme", ascending=self.ascending)
            else:
                df_sorted = df_sorted.sort_values(
                    self.annotator.column_name, ascending=self.ascending
                )
            return (df_sorted.index.values, None)

        def get_dependencies(self, gr, lanes):
            return [gr.load(), gr.add_annotator(self.annotator)], (lanes[0].name,)

    class OrderIthLaneSum(_OrderIthLane):
        """Order by the sum(signal strength) in the ith lane.
        @i may be either an integer index into the list of lanes_to_draw
        or or a AlignedLane object (which needs to be in lanes_to_draw!)
        Returns the indices of the order, and an (optional) cluster number - None for no clustering
        """

        def __init__(self, i):
            _OrderIthLane.__init__(self, i, lambda values: values.sum(axis=1), "sum")

    class OrderIthLaneMax(_OrderIthLane):
        """Order by the max(signal strength) in the ith lane.
        @i may be either an integer index into the list of lanes_to_draw
        or or a AlignedLane object (which needs to be in lanes_to_draw!)
        Returns the indices of the order, and an (optional) cluster number - None for no clustering
        """

        def __init__(self, i):
            _OrderIthLane.__init__(self, i, lambda values: values.max(axis=1), "max")

    class _OrderKMeans:
        def do_cluster(
            self, for_clustering, gr, lanes, raw_data, norm_data, seed=(1000, 2000)
        ):
            import scipy.cluster.vq
            import random

            random.seed(seed)
            np.random.seed(seed)
            for_clustering_whitened = scipy.cluster.vq.whiten(for_clustering)
            del for_clustering
            if self.no_of_clusters is None:
                no_of_clusters_to_use = 2 ** len(lanes)
            else:
                no_of_clusters_to_use = self.no_of_clusters
            while no_of_clusters_to_use > 0:  # retry with fewer clusters if it
                try:
                    codebook, distortion = scipy.cluster.vq.kmeans(
                        for_clustering_whitened, no_of_clusters_to_use
                    )
                    labels, distortion = scipy.cluster.vq.vq(
                        for_clustering_whitened, codebook
                    )
                    break
                except np.linalg.linalg.LinAlgError:
                    no_of_clusters_to_use -= 1
            del for_clustering_whitened

            first_lane_region_intensity_sum = norm_data[
                lanes[self.lane_to_sort_by].name
            ].max(axis=1)
            region_no_to_cluster = {}
            tuples_for_sorting = []
            for region_no, cluster_id in enumerate(labels):
                region_no_to_cluster[region_no] = cluster_id
                tuples_for_sorting.append(
                    (cluster_id, first_lane_region_intensity_sum[region_no], region_no)
                )
            tuples_for_sorting.sort(reverse=False)
            region_order = np.zeros(len(tuples_for_sorting), dtype=np.uint32)
            cluster_id_in_order = []
            for yy_value, tup in enumerate(tuples_for_sorting):
                cluster_id, donotcare, region_no = tup
                region_order[yy_value] = region_no
                cluster_id_in_order.append(cluster_id)
            return region_order, cluster_id_in_order

        def get_dependencies(self, gr, lanes):
            return (
                [
                    gr.load(),
                    ppg.FunctionInvariant(
                        "genomics.regions.heatmap._OrderKMeans.do_cluster",
                        self.__class__.do_cluster,
                    ),
                ],
                (self.lane_to_sort_by,),
            )

    class OrderClusterKMeans(_OrderKMeans):
        def __init__(
            self, no_of_clusters=None, lane_to_sort_by=0, lanes_to_include=None
        ):
            """Within a cluster, sort by lane_to_sort_by sum normalized signal"""
            if no_of_clusters is not None and not isinstance(no_of_clusters, int):
                raise ValueError("Invalid no_of_clusters")
            if not isinstance(lane_to_sort_by, int):
                raise ValueError("@lane_to_sort_by must be an integer")
            if lanes_to_include is not None:
                lane_str = ",".join([x.name for x in lanes_to_include])
            else:
                lane_str = "None"
            self.name = "OrderClusterKMeans_%s_%s" % (no_of_clusters, lane_str)
            self.no_of_clusters = no_of_clusters
            self.lane_to_sort_by = lane_to_sort_by
            self.lanes_to_include = lanes_to_include

        def calc(self, gr, lanes, raw_data, norm_data):
            row_count = raw_data[lanes[0].name].shape[0]
            if self.lanes_to_include is not None:
                cluster_lanes = self.lanes_to_include
            else:
                cluster_lanes = lanes
            for_clustering = np.empty((row_count, len(cluster_lanes)))
            for lane_no, lane in enumerate(cluster_lanes):
                lane_data = raw_data[lane.name]
                lane_data = lane_data.sum(axis=1)
                lane_data = lane_data - lane_data.min()  # norm to  0..1
                lane_data = lane_data / lane_data.max()
                for_clustering[:, lane_no] = lane_data

            return self.do_cluster(for_clustering, gr, lanes, raw_data, norm_data)

    class OrderClusterKMeans_ClusterSortedSignalCompatible(_OrderKMeans):
        """A reimplementation of the ClusterSortedSignal plot clustering"""

        def __init__(self, no_of_clusters=None):
            if no_of_clusters is not None and not isinstance(no_of_clusters, int):
                raise ValueError("Invalid no_of_clusters")
            self.name = "OrderClusterKMeans_CSSC_%s" % no_of_clusters
            self.lane_to_sort_by = 0
            self.no_of_clusters = no_of_clusters

        def calc(self, gr, lanes, raw_data, norm_data):
            row_count = raw_data[lanes[0].name].shape[0]
            vector_length = raw_data[lanes[0].name].shape[1]
            lane_count = len(lanes)
            for_clustering = np.empty((row_count, vector_length * lane_count))
            for lane_no, lane in enumerate(lanes):
                lane_data = raw_data[lane.name]
                offset = vector_length * lane_no
                for_clustering[:, offset : offset + vector_length] = lane_data
            return self.do_cluster(for_clustering, gr, lanes, raw_data, norm_data)

    class OrderCombinatorics:
        """A heatmap that ranks the conditions in each peak,
        than orders by the rank tuples (so first all peaks where condition 0 is the strongest,
        within that first all where condition 1 is the 2nd strongest, and so on)
        Experimental, usefulness in question
        """

        name = "OrderCombinatorics"

        def calc(self, gr, lanes, raw_data, norm_data):
            _apply_tpm(lanes, raw_data)
            for_clustering = {}
            for lane_no, lane in enumerate(lanes):
                lane_data = raw_data[lane.name].sum(axis=1)
                lane_data -= lane_data.min()  # so each lane is from 0..1
                lane_data /= lane_data.max()
                for_clustering[lane.name] = lane_data / len(
                    lanes
                )  # make them go from 0.. 1/lane_count
            df = pd.DataFrame(for_clustering)
            df += df.rank(
                axis=1
            )  # Ranks start at 1. So most siginficant digits are the ranks, then within one rank, sort by intensity of the respective lane...
            df = df.sort_values(list(df.columns[:-1]))
            order = list(df.index)
            dots = []
            last_key = False
            current_dot = 0
            for ii, pos in enumerate(order):
                key = tuple(df.iloc[pos].astype(int))
                if last_key is None or (key != last_key):
                    if current_dot:
                        current_dot = 0
                    else:
                        current_dot = 1
                    last_key = key
                dots.append(current_dot)
            return (order, dots)

        def get_dependencies(self, gr, lanes):
            return [gr.load()], (None,)

    class OrderStealFromOtherHeatmapPlot:
        """Copy the ordering from another heatmap.plot() call"""

        def __init__(self, other_heatmap_plot):
            self.name = (
                "StealFromOtherHeatmap" + other_heatmap_plot.heatmap.gr_to_draw.name
            )
            if not isinstance(other_heatmap_plot, _HeatmapPlot):
                raise ValueError(
                    "@other_heatmap_plot must be thue result of a Heatmap.plot call"
                )
            self.other_heatmap_plot = other_heatmap_plot

        def calc(self, gr, lanes, raw_data, norm_data):
            return self.other_heatmap_plot.order_

        def get_dependencies(self, gr, lanes):
            return [gr.load(), self.other_heatmap_plot.calc_order()], (None,)

    class OrderOverlappingClustersBySignalSumIthLane:
        """
        Create a cluster that overlaps with your gr_to_overlap and a cluster that does not overlap. And sort each cluster individually by the
        """

        def __init__(self, gr_to_overlap, lane_to_sort_by):
            if not isinstance(lane_to_sort_by, int):
                raise ValueError("@lane_to_sort_by must be an integer")
            self.name = "OrderOverlappingClustersBySignalSum_%s_th_lane" % str(
                lane_to_sort_by
            )
            self.lane_to_sort_by = lane_to_sort_by
            self.gr_to_overlap = gr_to_overlap

        def calc(self, gr, lanes, raw_data, norm_data):
            lane_name = lanes[self.lane_to_sort_by].name
            values = raw_data[lane_name]
            sums = values.sum(axis=1)
            out_list = []

            for index_row, my_sum in zip(
                gr.df[["chr", "start", "stop"]].iterrows(), sums
            ):
                if self.gr_to_overlap.has_overlapping(
                    index_row[1]["chr"], index_row[1]["start"], index_row[1]["stop"]
                ):
                    cluster_id = 1
                else:
                    cluster_id = 0
                out_list.append(
                    (cluster_id, my_sum, index_row[0])
                )  # index_row[0] is needed to identify the orignal row in the GenomicRegion the values belong to
            sorted_tuples = sorted(out_list, reverse=False)
            """
            eine calc(self, gr, lanes, raw_data, norm_data), und gibt zurueck: [ Reihenfolge in der die Regions aus GR gemalt werden sollen, Cluster-no], und jede clusterno wird nachher eine farbe
            """
            # first_lane_region_intensity_sum = norm_data[lanes[self.lane_to_sort_by].name].max(axis=1)

            return ([x[2] for x in sorted_tuples], [x[0] for x in sorted_tuples])

        def get_dependencies(self, gr, lanes):
            return (
                [
                    gr.load(),
                    gr.build_intervals(),
                    self.gr_to_overlap.load(),
                    self.gr_to_overlap.build_intervals(),
                ],
                (None,),
            )
