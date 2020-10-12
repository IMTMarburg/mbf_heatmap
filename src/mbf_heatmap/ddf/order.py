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
