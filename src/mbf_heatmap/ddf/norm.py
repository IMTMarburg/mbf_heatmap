import pandas as pd
import numpy as np


class NormStrategy:
    pass


class Unchanged(NormStrategy):
    """straight pass through"""

    name = "Unchanged"

    def calc(self, df, columns):
        return df[columns]

    def deps(self):
        return []


class Log2(NormStrategy):
    """straight pass through"""

    name = "Unchanged"

    def calc(self, df, columns):
        return pd.DataFrame({x: np.log2(df[x].values) for x in columns}, index=df.index)

    def deps(self):
        return []
