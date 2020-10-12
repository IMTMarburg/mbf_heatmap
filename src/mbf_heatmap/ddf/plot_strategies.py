import math


class Plot_Matplotlib:
    # much faster than Plot_GGPlot...
    """Allowed plot options:
    height
    max_width
    color_scale (from plt.cm, or a dictionary lane->plt.cm.*)
    hide_legend
    """

    name = "Plot_matplotlib"

    def plot(self, df, names_in_order, plot_options, title=""):
        import matplotlib.pyplot as plt
        import matplotlib

        plt.switch_backend("agg")

        cluster_ids = df["cluster"]
        df = df.drop("cluster", axis=1)
        row_count, col_count = df.shape
        plot_height = plot_options.get("height", max(5, math.ceil(row_count / 250.0)))
        plot_width = max(plot_options.get("max_width", 4), 1.55 * col_count) * 2
        if plot_options.get("color_scale", False) is False:
            color_scale = plt.cm.OrRd
        else:
            color_scale = plot_options["color_scale"]
        fig = plt.figure(figsize=(plot_width, plot_height))
        title = plot_options.get("title", "")
        if title:
            plt.suptitle(title)
        for ii, column in enumerate(df.columns):
            ax = fig.add_subplot(
                1, col_count + 1, ii + 1
            )  # matplot lib numbers subplots from 1!
            ax.set_title(names_in_order[ii])
            if plot_options.get("show_clusters", False):
                plt.scatter(
                    x=[-10] * row_count,
                    y=list(range(row_count)),
                    c=cluster_ids,
                    cmap=plt.cm.Set1,
                    s=100,
                    edgecolors="face",
                )
                pass
            data = df[[column]].values
            if isinstance(color_scale, dict):
                cs = color_scale[column.name]
            elif hasattr(color_scale, "__call__") and not isinstance(
                color_scale, matplotlib.colors.Colormap
            ):
                cs = color_scale(column)
            else:
                cs = color_scale
            plt.imshow(data, cmap=cs, interpolation="nearest", aspect="auto")
            plt.axis("off")

        if not plot_options.get("hide_legend", False):
            plt.colorbar()
        return plt

    def render(self, output_filename, p):
        p.savefig(output_filename, bbox_inches="tight")

    def get_parameters(self, plot_options, lanes_to_draw):
        cs = plot_options.get("color_scale", False)
        color_scale_string = color_scale_to_string(cs, lanes_to_draw)
        return (
            plot_options.get("height", None),
            plot_options.get("max_width", 4),
            color_scale_string,
            plot_options.get("hide_legend", False),
            plot_options.get("title", None),
        )

    def get_dependencies(self, heatmapplot, plot_options):
        return None
        # res = []
        # cs = plot_options.get("color_scale", False)
        # if hasattr(cs, '__call__'):
        #    res.append(ppg.FunctionInvariant(heatmapplot.output_filename + '_color_scale', cs))
        # return res


def color_scale_to_string(cs, lanes_to_draw):
    """Convert a matplotlib color scale into a string for parameter recording"""
    if hasattr(cs, "name"):
        color_scale_string = cs.name  # matplotlib colors scales have no useful __str__
    elif isinstance(cs, dict):
        if set(cs.keys()) != set([x.name for x in lanes_to_draw]):
            raise ValueError(
                "Using dict color_scale, but did not provide values for all lanes"
            )
        color_scale_string = ""
        for x in cs:
            color_scale_string += x + "==="
        if hasattr(cs[x], "name"):
            color_scale_string += cs[x].name
        else:
            color_scale_string += str(cs[x])
        color_scale_string += ";;;"
    elif hasattr(cs, "__call__"):
        color_scale_string = "function"
    elif cs is False:
        color_scale_string = "default"
    else:
        raise ValueError(
            "Color option was not a matplotlib.pyplot.cm. class or a dict of such. Was %s:"
            % cs
        )
    return color_scale_string
