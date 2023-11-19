"""
Varying visualisation tools.
"""

import os
import pickle
import warnings
import math
import graphviz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd


def draw_net(
    config,
    genome,
    view=False,
    filename=None,
    node_names=None,
    show_disabled=True,
    prune_unused=False,
    node_colors=None,
    fmt="pdf",
    detailed=False,
    node_outputs=None,
    hebbians=None,
    hebbian_trial_index=0,
    draw_std=False,
):
    """Receives a genome and draws a neural network with arbitrary topology."""
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn(
            "This display is not available due to a missing optional dependency (graphviz)"
        )
        return

    # If requested, use a copy of the genome which omits all components that won't affect the output.
    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {"shape": "circle", "fontsize": "9", "height": "0.10", "width": "0.10"}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        if detailed and node_outputs:
            name = f"{name}\noutput {'%.3f' % node_outputs[k]}"
            node_names[k] = name
        input_attrs = {
            "style": "filled",
            "shape": "box",
            "fillcolor": node_colors.get(k, "lightgray"),
        }
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        if detailed:
            node = genome.nodes[k]
            name = f"{name}\nbias {'%.3f' % node.bias}\n&#945; {'%.3f' % node.response}"
            if node_outputs:
                name = name + f"\noutput {'%.3f' % node_outputs[k]}"
            node_names[k] = name
        node_attrs = {"style": "filled", "fillcolor": node_colors.get(k, "lightblue")}

        dot.node(name, _attributes=node_attrs)

    used_nodes = set(genome.nodes.keys())
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        name = node_names.get(n, str(n))
        if detailed:
            node = genome.nodes[n]
            name = f"key {name}\nbias {'%.3f' % node.bias}\n&#945; {'%.3f' % node.response}"
            if node_outputs:
                name = name + f"\noutput {'%.3f' % node_outputs[n]}"
            node_names[n] = name
        attrs = {"style": "filled", "fillcolor": node_colors.get(n, "white")}
        dot.node(name, _attributes=attrs)

    if draw_std:
        hebbian_std_df = get_hebbian_std(hebbian_to_dataframe(hebbians))

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = "solid" if cg.enabled else "dotted"
            color = "red" if cg.weight > 0 else "blue"
            width = "0.1"
            if draw_std:
                connection_std = hebbian_std_df.loc[
                    (hebbian_std_df["prior_node"] == int(input))
                    & (hebbian_std_df["posterior_node"] == int(output))  # VERIFY ORDER
                ]
                if connection_std.empty:
                    label = "    "
                else:
                    std = connection_std.iloc[0]["std"]
                    label = "{:.3f}".format(std)
                    width = str(0.1 + abs(std * 35))
            elif hebbians:
                label = "{:.3f}".format(
                    hebbians[hebbian_trial_index][0].get(output, {}).get(input, 0)
                    * genome.nodes[output].response
                    + cg.weight
                )
            else:
                label = "{:.3f}".format(cg.weight)
                width = str(0.1 + abs(cg.weight * 3.5))
            dot.edge(
                a,
                b,
                label=label,
                _attributes={
                    "style": style,
                    "color": color,
                    "penwidth": width,
                    "fontsize": "9",
                },
            )

    dot.render(filename, view=view)

    return dot


def onclick(event):
    """
    Click handler for weight gradient created by a CPPN. Will re-query with the clicked coordinate.
    """
    plt.close()
    x = event.xdata
    y = event.ydata

    path_to_cppn = "es_hyperneat_xor_small_cppn.pkl"
    # For now, path_to_cppn should match path in split_output_cppn.py, sorry.
    with open(path_to_cppn, "rb") as cppn_input:
        cppn = pickle.load(cppn_input)
        from pureples.es_hyperneat.es_hyperneat import find_pattern

        pattern = find_pattern(cppn, (x, y))
        draw_pattern(pattern)


def draw_pattern(im, res=60):
    """
    Draws the pattern/weight gradient queried by a CPPN.
    """
    fig = plt.figure()
    plt.axis([-1, 1, -1, 1])
    fig.add_subplot(111)

    a = range(res)
    b = range(res)

    for x in a:
        for y in b:
            px = -1.0 + (x / float(res)) * 2.0 + 1.0 / float(res)
            py = -1.0 + (y / float(res)) * 2.0 + 1.0 / float(res)
            c = str(0.5 - im[x][y] / float(res))
            plt.plot(px, py, marker="s", color=c)

    fig.canvas.mpl_connect("button_press_event", onclick)

    plt.grid()
    # plt.show()


def draw_es(id_to_coords, connections, filename):
    """
    Draw the net created by ES-HyperNEAT
    """
    fig = plt.figure()
    # plt.axis([-1.1, 1.1, -1.1, 1.1])
    axs = fig.add_subplot(111)
    axs.set_axisbelow(True)

    for c in connections:
        edge_color = "black"
        color = "red"
        shape = "left"
        if c.weight > 0.0:
            color = "blue"
            shape = "right"
        rad = f"rad=.0"
        # Adjust curvature of connection if it crosses a node or is recurrent
        for coord, _ in id_to_coords.items():
            if abs(
                math.dist(coord, [c.x1, c.y1])
                + math.dist(coord, [c.x2, c.y2])
                - math.dist([c.x1, c.y1], [c.x2, c.y2])
            ) < 0.01 and not (coord == (c.x1, c.y1) or coord == (c.x2, c.y2)):
                rad = f"rad=.1"
                break
            elif c.recurrent:
                rad = f"rad=.1"
                edge_color = "grey"
                break
        plt.gca().add_patch(
            patches.FancyArrowPatch(
                (c.x1, c.y1),
                (c.x2, c.y2),
                connectionstyle=f"arc3,{rad}",
                fc=color,
                ec=edge_color,
                arrowstyle="Simple, tail_width=0.8, head_width=2, head_length=4",
                linewidth=0.4,
                # length_includes_head=True,
                # head_width=0.03,
                # shape=shape,
                linestyle="-",
            )
        )

    for coord, _ in id_to_coords.items():
        plt.plot(coord[0], coord[1], marker="o", markersize=3.0, color="grey")

    plt.grid()
    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close()


# Function to divide the groups into chunks of size n and calculate the mean
def chunked_mean(group, chunk_size):
    # indices_to_keep = [i for i in range(len(group)) if i % 100 >= chunk_size]
    # # Calculate the number of chunks
    # group = group.iloc[indices_to_keep]
    num_chunks = len(group) // chunk_size
    # Reshape the array into num_chunks of chunk_size and calculate the mean of each
    return pd.Series(
        group[: num_chunks * chunk_size].values.reshape(-1, chunk_size).mean(axis=1)
    )


def get_hebbian_std(hebbian, group_size=10):
    std_data = []
    for (prior, posterior), group in hebbian.groupby(["prior_node", "posterior_node"]):
        chunk_means = chunked_mean(group["weight"], group_size)
        std_data.append(
            {
                "prior_node": prior,
                "posterior_node": posterior,
                "std": chunk_means.std(),
            }
        )

    return pd.DataFrame(std_data)


def hebbian_to_dataframe(hebbian):
    flattened_data = []
    for trial_index, trial in enumerate(hebbian):
        for node_dict in trial:
            for posterior_node, weights_dict in node_dict.items():
                for prior_node, weight in weights_dict.items():
                    flattened_data.append(
                        {
                            "trial": trial_index,  # Trial index
                            "prior_node": prior_node,
                            "posterior_node": posterior_node,
                            "weight": weight,
                        }
                    )

    return pd.DataFrame(flattened_data)


def draw_hebbian(
    net,
    filename,
    node_names=None,
):
    hebbian = net.hebbian_update_log
    fig = plt.figure()

    df = hebbian_to_dataframe(hebbian)
    df.to_csv(filename.split(".")[0] + ".csv")

    std = get_hebbian_std(df)
    filtered_columns = std[std["std"] > 0.001][["prior_node", "posterior_node"]]
    df = pd.merge(
        df, filtered_columns, on=["prior_node", "posterior_node"], how="inner"
    )
    node_df = pd.DataFrame(
        [(t, 0.0) for t in net.input_nodes] + [(t[0], t[4]) for t in net.node_evals],
        columns=["node", "response"],
    )
    outgoing_nodes = df["prior_node"].unique()
    outgoing_nodes = outgoing_nodes[np.argsort(np.abs(outgoing_nodes))]

    # Calculate the number of rows needed for a 3-column layout
    num_nodes = len(outgoing_nodes)
    num_rows = (
        num_nodes + 2
    ) // 3  # Add 2 to ensure that we have enough rows for all nodes
    # Create subplots in a 3xN grid
    if num_rows == 0:
        return
    fig, axes = plt.subplots(
        num_rows, 3, figsize=(15, num_rows * 5), sharey=True
    )  # Adjust the figsize as necessary
    # plt.setp(axes, ylim=(0, 1))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    for idx, node in enumerate(outgoing_nodes):
        # Plot each node in its respective subplot
        # axes[idx].plot(df.index, node_df[node], label=node)
        current_node = node
        if current_node in outgoing_nodes:
            axes[idx].set_title(
                node_names[current_node]
                if current_node in node_names
                else f"Node {current_node}"
            )
            axes[idx].set_xlabel("Trial")
            if idx % 3 == 0:
                axes[idx].set_ylabel("Magnitude")
            to_plot = df[df["prior_node"] == current_node].reset_index(drop=True)
            for posterior, group in to_plot.groupby(["posterior_node"]):
                axes[idx].plot(
                    group["trial"],
                    group["weight"]
                    * node_df[node_df["node"] == posterior]
                    .reset_index(drop=True)
                    .at[0, "response"],
                    label=posterior,
                )
            for x in [50, 100, 150, 200]:
                axes[idx].axvline(
                    x=x,
                    ymin=0.0,
                    ymax=1,
                    c="gray",
                    linewidth=1,
                    zorder=-1,
                    clip_on=False,
                )
            axes[idx].legend(loc="upper right")

    # If there are any empty subplots, turn them off
    for ax in axes[num_nodes:]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_hebbian_correlation_heatmap(csv_file_path, mapping_list, filename):
    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # Rename nodes -1, -2, and 0
    node_renaming = {-1: "ready", -2: "go", 0: "output"}
    df.replace(
        {"prior_node": node_renaming, "posterior_node": node_renaming}, inplace=True
    )

    # For every 50 trials, take the average of trials 10-50
    df["trial_group"] = df["trial"] // 50
    filtered_df = df[(df["trial"] % 50 >= 10) & (df["trial"] % 50 <= 49)]
    average_df = (
        filtered_df.groupby(["trial_group", "prior_node", "posterior_node"])
        .agg(avg_weight=("weight", "mean"))
        .reset_index()
    )

    # Map each trial group to an entry in the mapping list
    average_df["mapped_value"] = average_df["trial_group"].apply(
        lambda x: mapping_list[x % len(mapping_list)]
    )

    # Calculate the correlation for each pair of 'prior_node' and 'posterior_node'
    grouped = (
        average_df.groupby(["prior_node", "posterior_node"])
        .apply(lambda g: g["mapped_value"].corr(g["avg_weight"]))
        .reset_index(name="correlation")
    )
    grouped = grouped.dropna()

    # Create a pivot table for the heatmap
    pivot_table = grouped.pivot("prior_node", "posterior_node", "correlation")

    # Plotting the heatmap
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(
        pivot_table,
        cmap="coolwarm",
        annot=False,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Correlation"},
    )
    plt.xlabel("Posterior Node")
    plt.ylabel("Prior Node")
    fig.savefig(
        filename,
        dpi=300,
    )


def draw_omission_trials(omission_outputs, filename):
    fig = plt.figure()
    for index, outputs in enumerate(omission_outputs):
        plt.plot(outputs, label=f"foreperiod {index+1}")
    lgd = plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlabel("timestep")
    plt.ylabel("magnitude")
    fig.savefig(
        filename,
        dpi=300,
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )
    plt.close()


def draw_output(
    cycle_len,
    network_input,
    network_output,
    expected_output,
    filename,
    end_tests=0,
    all_outputs=[],
    network=None,
    draw_std=False,
):
    df = pd.DataFrame(
        {
            "input": network_input,
            "expected_output": expected_output,
            "network_output": network_output,
        },
    )
    df.to_csv(filename.split(".")[0] + ".csv")
    fig = plt.figure()
    indices = np.asarray(np.where(network_input == 1))[0][1:]

    split_output = np.array(np.split(network_output, indices), dtype=object)

    split_expected_output = np.array(np.split(expected_output, indices), dtype=object)
    split_expected_end_test_output = np.array(
        np.split(expected_output, indices)[-end_tests:], dtype=object
    )

    # Pad arrays
    split_output = np.array(
        [np.pad(i, ((0, cycle_len - len(i)))) for i in split_output]
    )
    split_expected_output = np.array(
        [np.pad(i, ((0, cycle_len - len(i)))) for i in split_expected_output]
    )

    # Calculate average for each timestep
    expected_avg = np.average(split_expected_output[6:-end_tests], axis=0)
    avg = np.average(split_output[6:-end_tests], axis=0)
    first = split_output[0]
    trained = split_output[6]
    last = split_output[-(end_tests + 1)]
    # max_out = np.max(split_output[30:-end_tests], axis=0)
    plt.plot(expected_avg, label="expected")
    # plt.plot(avg, label="average")
    # plt.plot(first, label="first")
    # plt.plot(trained, label="trained_early")
    plt.plot(last, label="trained")
    if end_tests:
        omission = split_output[-(end_tests)]
        plt.plot(omission, label="omission")
    # plt.plot(max_out, label="maximum")
    lgd = plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlabel("timestep")
    plt.ylabel("magnitude")
    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close()
    if end_tests:
        end_test_set = split_output[-end_tests:-1]
        fig2 = plt.figure()
        plt.plot(expected_avg, label="expected_avg")
        # plt.plot(end_test_set[0], label="first")
        # plt.plot(end_test_set[1], label="middle")
        # plt.plot(end_test_set[2], label="end")
        plt.plot(end_test_set[0], label="omission")
        lgd = plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.xlabel("timestep")
        plt.ylabel("output magnitude")
        fig2.savefig(
            filename.split(".")[0] + "_end_tests.png",
            dpi=300,
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )
        plt.close()
    if draw_std:
        draw_net(
            network["config"],
            network["genome"],
            detailed=True,
            filename=filename.split(".")[0] + "_hebbian_std_network",
            hebbians=network["hebbians"],
            node_names={
                -1: "ready",
                -2: "go",
                0: "output",
                # 1: "output2",
            },
            node_colors={
                -1: "yellow",
                -2: "green",
                0: "lightblue",
                # 1: "lightblue",
            },
            prune_unused=True,
            draw_std=draw_std,
        )
    if len(all_outputs):
        df = pd.json_normalize(all_outputs)
        df.to_csv(filename.split(".")[0] + f"_all.csv")
        test_names = [
            "trial_last",
            "go_omitted",
            "normalizing_trial",
            "trial_first",
            "trial_second",
            "trial_third",
            "go_on_first",
            "go_on_middle",
            "go_on_end",
        ]
        indices_from_zero = np.append(0, indices)
        for test in range(end_tests + 4):
            indice_index = -(end_tests - test) - 1
            start = indices_from_zero[indice_index]
            end = indices_from_zero[indice_index + 1] if end_tests - test else None
            df = pd.json_normalize(all_outputs[start:end])
            # print("test ", test)
            # print("end_tests ", end_tests)
            # print("indices_from_zero ", indices_from_zero)
            # print(indices_from_zero[indice_index])
            # print("start ", start)
            # print("end ", end)
            # print(df)
            named_df = df.rename(columns={"-2": "ready", "-1": "go", "0": "output"})
            fig3 = plt.figure()
            plt.plot(
                df.iloc[:, 0],
                linestyle="--",
                color="y",
            )
            plt.plot(
                df.iloc[:, 1],
                linestyle="--",
                color="green",
            )
            plt.plot(
                df.iloc[:, 2],
                linestyle="-.",
                color="grey",
            )
            plt.plot(df.iloc[:, 3:])
            lgd = plt.legend(
                named_df.columns, bbox_to_anchor=(1.04, 1), borderaxespad=0
            )
            fig3.savefig(
                filename.split(".")[0] + f"_all_{test_names[test]}.png",
                dpi=300,
                bbox_extra_artists=(lgd,),
                bbox_inches="tight",
            )
            plt.close()
            if network and test_names[test] in [
                "trial_last",
                # "trial_first",
                "go_omitted",
            ]:
                save_dir = (
                    filename.split(".")[0]
                    + f"_all_{test_names[test]}_detailed_network/"
                )
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                hebbian_trial_index = (
                    len(network["hebbians"]) - len(indices_from_zero)
                    if test_names[test] == "trial_first"
                    else -1
                )
                for timestep in range(len(df.iloc[:, :])):
                    draw_net(
                        network["config"],
                        network["genome"],
                        filename=save_dir + f"timestep_{timestep}",
                        hebbians=network["hebbians"],
                        hebbian_trial_index=hebbian_trial_index,
                        node_outputs=df.iloc[timestep, :].to_dict(),
                        node_names={
                            -1: "ready",
                            -2: "go",
                            0: "output",
                            # 1: "output2",
                        },
                        node_colors={
                            -1: "yellow",
                            -2: "green",
                            0: "lightblue",
                            # 1: "lightblue",
                        },
                        prune_unused=True,
                        detailed=True,
                    )
    return


def draw_tests(
    cycle_len,
    network_input,
    network_output,
    expected_output,
    filename,
):
    return
