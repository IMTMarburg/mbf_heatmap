class NormStrategy:
    pass


class Unchanged(NormStrategy):
    """straight pass through"""

    name = "Unchanged"

    def calc(self, df, columns):
        return df[columns]

    def deps(self):
        return []
