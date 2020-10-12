import math
import pandas as pd


class Plot_Matplotlib:
    # much faster than Plot_GGPlot...
    """Allowed plot options:
    height
    max_width
    title
    color_scale (from plt.cm, or a dictionary lane->plt.cm.*)
    hide_legend
    """

    name = "Plot_matplotlib"

    def plot(
        self,
        gr_to_draw,
        lanes_to_draw,
        raw_data,
        norm_data,
        order,
        names_in_order,
        plot_options,
    ):
        import matplotlib.pyplot as plt
        import matplotlib

        plt.switch_backend("agg")
        seen = set()
        for lane in lanes_to_draw:
            if lane.name in seen:
                raise ValueError("Duplicate lanes in lanes to draw: %s" % lane.name)
            seen.add(lane.name)
        peak_count = len(list(norm_data.values())[0])
        lane_count = len(norm_data)
        if "height" not in plot_options:
            plot_height = max(5, math.ceil(peak_count / 250.0))
        else:
            plot_height = plot_options["height"]
        plot_width = (
            max(plot_options.get("max_width", 4), 1.55 * len(lanes_to_draw)) * 2
        )
        peak_order = order[0][::-1]
        cluster_order = order[1]
        if cluster_order is not None:
            cluster_order = order[1][::-1]
        if plot_options.get("color_scale", False) is False:
            color_scale = plt.cm.OrRd
        else:
            color_scale = plot_options["color_scale"]
        fig = plt.figure(figsize=(plot_width, plot_height))
        title = plot_options.get("title", True)
        if title is True:
            title = "%i regions from %s" % (peak_count, gr_to_draw.name)
        elif title:
            title = title.replace("%i", "%i" % peak_count).replace(
                "%s", gr_to_draw.name
            )
        if title is not None:
            plt.suptitle(title)
        for ii, lane in enumerate(lanes_to_draw):
            lane_name = lane.name
            ax = fig.add_subplot(
                1, lane_count, ii + 1
            )  # matplot lib numbers subplots from 1!
            ax.set_title(names_in_order[ii])
            if cluster_order is not None:
                plt.scatter(
                    x=[-10] * peak_count,
                    y=list(range(peak_count)),
                    c=cluster_order,
                    cmap=plt.cm.Set1,
                    s=100,
                    edgecolors="face",
                )
                pass
            data = norm_data[lane_name][peak_order]
            if isinstance(color_scale, dict):
                cs = color_scale[lane.name]
            elif hasattr(color_scale, "__call__") and not isinstance(
                color_scale, matplotlib.colors.Colormap
            ):
                cs = color_scale(lane)
            else:
                cs = color_scale
            plt.imshow(data, cmap=cs, interpolation="nearest", aspect="auto")
            plt.axis("off")

        if not plot_options.get("hide_legend", False):
            plt.colorbar()
        if plot_options.get("dump_df", False):
            """ Makes a Dataframe and writes this - dataframe should be about the GenomicRegion (gr_to_draw)"""
            df_dump_filename = plot_options["dump_df"]
            if not isinstance(df_dump_filename, str):
                raise ValueError(
                    "dump_df needs a string that specifies the outputpath and filename"
                )
            out_dict = {"chr": [], "start": [], "stop": [], "cluster_id": []}
            import itertools

            gr_to_draw_df = gr_to_draw.df
            for x, y in itertools.izip(order[0], order[1]):
                out_dict["start"].append(gr_to_draw_df["start"][x])
                out_dict["stop"].append(gr_to_draw_df["stop"][x])
                out_dict["chr"].append(gr_to_draw_df["chr"][x])
                out_dict["cluster_id"].append(y)
            df = pd.DataFrame(out_dict)
            df.to_csv(df_dump_filename, sep="\t", index=False)
        return plt

    def render(self, output_filename, p):
        p.savefig(output_filename, bbox_inches="tight")

    def get_parameters(self, plot_options, lanes_to_draw):
        cs = plot_options.get("color_scale", False)
        if hasattr(cs, "name"):
            color_scale_string = (
                cs.name
            )  # matplotlib colors scales have no useful __str__
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
        return (
            plot_options.get("height", None),
            plot_options.get("max_width", 4),
            color_scale_string,
            plot_options.get("hide_legend", False),
            plot_options.get("title", None),
        )

    def get_dependencies(self, heatmapplot, plot_options):
        return None
