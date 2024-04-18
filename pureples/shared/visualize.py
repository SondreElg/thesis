"""
Varying visualisation tools.
"""

import os
import pickle
import warnings
import math
import graphviz
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd

standard_colors = [
    "blue",
    "green",
    "red",
    "purple",
    "orange",
    "yellow",
    "pink",
    "brown",
    "gray",
    "olive",
    "cyan",
    "lime",
    "teal",
    "lavender",
    "magenta",
    "salmon",
    "gold",
    "lightblue",
    "lightgreen",
    "lightcoral",
    "lightpink",
    "lightgray",
    "lightyellow",
    "lightbrown",
    "lightolive",
    "lightcyan",
    "lightlime",
    "lightteal",
    "lightlavender",
    "lightmagenta",
    "lightsalmon",
    "lightgold",
]


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


def make_genome_matrix(
    config,
    genome,
    filename=None,
    prune_unused=False,
    node_names={
        -1: "ready",
        -2: "go",
        -3: "bias",
        0: "output",
        # 1: "output2",
    },
):
    # If requested, use a copy of the genome which omits all components that won't affect the output.
    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    # nodes = set(*genome.nodes.keys(), *config.genome_config.output_keys)

    connections = [(cg.key, cg.weight) for cg in genome.connections.values()]

    # Extracting unique nodes and sorting them
    nodes = set()
    for (a, b), _ in connections:
        nodes.update([a, b])
    nodes = sorted(nodes)
    bias_row = pd.DataFrame(
        [genome.nodes[k].bias if k in genome.nodes else 0.0 for k in nodes],
        index=nodes,
        columns=["-3"],
    ).T
    # Creating an empty DataFrame with sorted nodes as index and columns
    matrix = pd.DataFrame(index=nodes, columns=nodes).fillna(0.0)

    matrix = pd.concat([bias_row, matrix])
    # Populating the DataFrame with weights as floats
    for (a, b), weight in connections:
        matrix.at[a, b] = weight

    # Save the DataFrame
    matrix.to_csv(f"{filename}.csv")


def onclick(event):
    """
    Click handler for weight gradient created by a CPPN. Will re-query with the clicked coordinate.
    """
    plt.close("all")
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
    plt.close("all")


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

    unfiltered_df = hebbian_to_dataframe(hebbian)

    std = get_hebbian_std(unfiltered_df)
    filtered_columns = std[std["std"] != 0.0][["prior_node", "posterior_node"]]
    df = pd.merge(
        unfiltered_df,
        filtered_columns,
        on=["prior_node", "posterior_node"],
        how="inner",
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

    # Don't plot if there are no Hebbian changes
    if num_rows == 0:
        return
    unfiltered_df.to_csv(filename.split(".")[0] + ".csv")

    # Create subplots in a 3xN grid
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
    plt.close("all")


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
    plt.close("all")


def pad_output_delays(
    cycle_len,
    cycle_delay_max,
    all_outputs,
    trials,
):
    max_foreperiod = cycle_len - cycle_delay_max
    dim = np.arange(-cycle_len, cycle_len, 1)

    indices = np.empty((len(all_outputs)), dtype=object)
    nodes = all_outputs[0][0].keys()
    padded_output = np.empty_like(all_outputs)
    delay_split = np.empty_like(all_outputs)
    for idx, node in enumerate(nodes):
        for block in range(len(all_outputs)):
            delay_split[block][node] = np.empty((cycle_delay_max), dtype=object)
            node_outputs = np.array(
                [entry[node] for entry in all_outputs[block]], dtype=float
            )
            if node == -1:
                indices[block] = np.append(
                    np.asarray(np.where(node_outputs == 1))[0], len(node_outputs)
                )

            block_indices = indices[block]
            entries = np.array([])
            delay_split_buffer = np.empty((cycle_delay_max), dtype=object)
            for i in range(len(block_indices) - 1):
                # print(f"{block_indices[i]}:{block_indices[i+1]}")
                first_half = np.pad(
                    node_outputs[block_indices[i] : block_indices[i + 1]],
                    (
                        0,
                        cycle_len
                        - len(node_outputs[block_indices[i] : block_indices[i + 1]]),
                    ),
                    constant_values=(np.nan,),
                )
                delay = block_indices[i + 1] - block_indices[i] - (max_foreperiod + 1)
                entries.extend(first_half)
                delay_split_buffer[delay].extend(first_half)
            delay_split[block][node][delay] = entries
            padded_output[block][node] = first_half
    return padded_output, delay_split


def process_output(
    cycle_len,
    cycle_delay_max,
    all_outputs,
    trials,
    *,
    include_previous=False,
    only_last=False,
    only_last_of_previous=False,
    custom_range=None,
    last_max_delay=False,
    end_tests=0,
    shape=("delay_block", "node_count", "foreperiod_blocks", "trials", "max_trial_len"),
):
    shape_map = {
        "delay_block": 0,
        "node_count": 1,
        "foreperiod_blocks": 2,
        "trials": 3,
        "max_trial_len": 4,
    }
    max_foreperiod = cycle_len - cycle_delay_max
    only_last = only_last or last_max_delay
    indices = np.empty((len(all_outputs)), dtype=object)
    dim = np.arange(-1 if only_last_of_previous else -cycle_len, cycle_len, 1)
    # dim = np.append(np.arange(1, cycle_len), np.arange(cycle_len))

    nodes = all_outputs[0][0].keys()

    # Calculate the number of rows needed for a 3-column layout
    num_nodes = len(nodes)

    # shape: (delay_block, node_count, foreperiod_blocks, trials, max_trial_len)
    delay_split = np.empty(
        (cycle_delay_max, num_nodes, len(all_outputs), trials, len(dim)), dtype=float
    )
    delay_split.fill(np.nan)

    #  Plot each node in its respective subplot
    previous_output = [
        None for _ in nodes
    ]  # NEED TO HAVE AN ENTRY FOR EACH NODE IN THIS ARRAY
    for block in range(len(all_outputs)):
        for idx, node in enumerate(nodes):
            node_outputs = np.array(
                [entry[node] for entry in all_outputs[block]], dtype=float
            )
            if node == -1:
                indices[block] = np.append(
                    np.asarray(np.where(node_outputs == 1))[0], len(node_outputs)
                )
                if end_tests:
                    indices[block] = indices[block][0:-(end_tests)]
                if only_last:
                    if last_max_delay:
                        for index in range(-1, -len(indices) + 2, -1):
                            if (
                                indices[block][index - 2] - indices[block][index]
                                == cycle_len * 2 + 1
                            ):
                                indices[block] = indices[block][index - 2 : index]
                                break
                    if indices[block][-1] - indices[block][0] > cycle_len * 2:
                        indices[block] = indices[block][-3:]
                if custom_range:
                    indices[block] = indices[block][custom_range[0] : custom_range[1]]

            block_indices = indices[block]
            entries = previous_output[idx] if previous_output[idx] is not None else []
            if only_last_of_previous:
                entries.append(
                    np.concatenate(
                        (
                            [0],
                            np.pad(
                                node_outputs[block_indices[0] : block_indices[1]],
                                (
                                    0,
                                    cycle_len
                                    - len(
                                        node_outputs[
                                            block_indices[0] : block_indices[1]
                                        ]
                                    ),
                                ),
                                constant_values=(np.nan,),
                            ),
                        )
                    )
                )
            # print(f"{previous_output=}")
            # print(f"{entries=}")
            for i in range(len(block_indices) - 2):
                # print(f"{block_indices[i]}:{block_indices[i+1]}")

                if only_last_of_previous:
                    first_half = node_outputs[block_indices[i]]
                else:
                    first_half = np.pad(
                        node_outputs[block_indices[i] : block_indices[i + 1]],
                        (
                            0,
                            cycle_len
                            - len(
                                node_outputs[block_indices[i] : block_indices[i + 1]]
                            ),
                        ),
                        constant_values=(np.nan,),
                    )
                second_half = np.pad(
                    node_outputs[block_indices[i + 1] : block_indices[i + 2]],
                    (
                        0,
                        cycle_len
                        - len(
                            node_outputs[block_indices[i + 1] : block_indices[i + 2]]
                        ),
                    ),
                    constant_values=(np.nan,),
                )
                # print(delay_split)
                delay_first = (
                    block_indices[i + 1] - block_indices[i] - (max_foreperiod + 1)
                )
                delay_split[delay_first][idx][block][i] = np.append(
                    first_half, second_half
                )
            if include_previous:
                previous_output[idx] = [np.append(first_half, second_half)]
    return np.moveaxis(delay_split, [0, 1, 2, 3, 4], [shape_map[x] for x in shape])


def draw_average_node_output(
    cycle_len,
    cycle_delay_max,
    network_input,
    all_outputs,
    filename,
    trials,
    *,
    only_last=False,
    only_last_of_previous=False,
    custom_range=None,
    last_max_delay=False,
    end_tests=0,
    log_level=0,
    node_names={
        -1: "ready",
        -2: "go",
        0: "output",
        # 1: "output2",
    },
):
    max_foreperiod = cycle_len - cycle_delay_max
    only_last = only_last or last_max_delay
    indices = np.empty((len(all_outputs)), dtype=object)
    dim = np.arange(-1 if only_last_of_previous else -cycle_len, cycle_len, 1)

    delay_split = np.empty(
        (cycle_delay_max, len(all_outputs), trials, len(dim)), dtype=float
    )
    delay_split.fill(np.nan)

    nodes = all_outputs[0][0].keys()

    # Calculate the number of rows needed for a 3-column layout
    num_nodes = len(nodes)
    num_rows = (
        num_nodes + 2 + 3
    ) // 3  # Add 2 to ensure that we have enough rows for all nodes

    # Create subplots in a 3xN grid
    fig, axes = plt.subplots(
        num_rows, 3, figsize=(15, num_rows * 5 + 1), sharey=True
    )  # Adjust the figsize as necessary

    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    delay = 0
    #  Plot each node in its respective subplot
    for idx, node in enumerate(nodes):
        axes[idx].set_title(node_names[node] if node in node_names else f"Node {node}")
        axes[idx].set_xlabel("Timestep")
        axes[idx].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        if idx % 3 == 0:
            axes[idx].set_ylabel("Magnitude")
        for x in [
            0 if only_last_of_previous else -cycle_delay_max,
            0,
            max_foreperiod,
        ]:
            axes[idx].axvline(
                x=x,
                ymin=0.0,
                ymax=1,
                c="gray",
                linewidth=1,
                linestyle="-" if x == 0 else "--",
                zorder=-1,
                clip_on=False,
            )
        for block in range(len(all_outputs)):
            node_outputs = np.array(
                [entry[node] for entry in all_outputs[block]], dtype=float
            )
            if node == -1:
                indices[block] = np.append(
                    np.asarray(np.where(node_outputs == 1))[0], len(node_outputs)
                )
                if end_tests:
                    indices[block] = indices[block][0:-(end_tests)]
                if only_last:
                    if last_max_delay:
                        for index in range(-1, -len(indices) + 2, -1):
                            if (
                                indices[block][index - 2] - indices[block][index]
                                == cycle_len * 2 + 1
                            ):
                                indices[block] = indices[block][index - 2 : index]
                                break
                    if indices[block][-1] - indices[block][0] > cycle_len * 2:
                        indices[block] = indices[block][-3:]
                if custom_range:
                    indices[block] = indices[block][custom_range[0] : custom_range[1]]

            block_indices = indices[block]
            entries = []
            if only_last_of_previous:
                entries.append(
                    np.concatenate(
                        (
                            [0],
                            np.pad(
                                node_outputs[block_indices[0] : block_indices[1]],
                                (
                                    0,
                                    cycle_len
                                    - len(
                                        node_outputs[
                                            block_indices[0] : block_indices[1]
                                        ]
                                    ),
                                ),
                                constant_values=(np.nan,),
                            ),
                        )
                    )
                )
            for i in range(len(block_indices) - 2):
                # print(f"{block_indices[i]}:{block_indices[i+1]}")

                if only_last_of_previous:
                    first_half = node_outputs[block_indices[i]]
                else:
                    first_half = np.pad(
                        node_outputs[block_indices[i] : block_indices[i + 1]],
                        (
                            0,
                            cycle_len
                            - len(
                                node_outputs[block_indices[i] : block_indices[i + 1]]
                            ),
                        ),
                        constant_values=(np.nan,),
                    )
                second_half = np.pad(
                    node_outputs[block_indices[i + 1] : block_indices[i + 2]],
                    (
                        0,
                        cycle_len
                        - len(
                            node_outputs[block_indices[i + 1] : block_indices[i + 2]]
                        ),
                    ),
                    constant_values=(np.nan,),
                )
                # print(delay_split)
                delay_first = (
                    block_indices[i + 1] - block_indices[i] - (max_foreperiod + 1)
                )
                delay_split[delay_first][block][i] = np.append(first_half, second_half)
                entries.append(np.append(first_half, second_half))
            entries = np.array(entries)
            latest_plot = axes[idx].plot(
                dim,
                entries[0] if only_last else np.nanmean(entries, axis=0),
                label=block + 1,
            )
            if not only_last:
                # print(
                #     node_names[node] if node in node_names else f"Node {node}",
                #     "block ",
                #     block,
                # )
                delay = 0
                for delayed_block in delay_split:
                    averages = np.nanmean(delayed_block[block], axis=0)
                    averages[cycle_len - 1] = averages[max_foreperiod + delay]
                    averages[cycle_len * 2 - 1] = averages[
                        cycle_len + max_foreperiod + delay
                    ]
                    mask = np.concatenate(
                        (
                            np.full(max_foreperiod, False),
                            np.isfinite(averages[max_foreperiod:cycle_len]),
                            np.full(max_foreperiod, True),
                            np.isfinite(averages[max_foreperiod + cycle_len :]),
                        ),
                        axis=None,
                    )
                    # print(mask, "len", len(mask))
                    # print(dim[mask])
                    # print(latest_plot)
                    # print("delay: ", delay, averages)
                    # print(averages)
                    # print("delay index", -(cycle_delay_max - delay))
                    # print(cycle_len - (cycle_delay_max - delay))
                    axes[idx].plot(
                        dim[mask],
                        averages[mask],
                        marker=f"{(delay % 4) + 1}",
                        markersize=5.0,
                        linewidth=0.3,
                        # linestyle="",
                        color=latest_plot[0].get_color(),
                        alpha=0.7,
                    )
                    # if delay < len(delay_split) - 1:
                    #     axes[idx].hlines(
                    #         xmin=-(cycle_delay_max - delay),
                    #         xmax=-1,
                    #         y=averages[cycle_len - (cycle_delay_max - delay)],
                    #         color=latest_plot[0].get_color(),
                    #         linewidth=0.5,
                    #         linestyle="--",
                    #         zorder=-1,
                    #         clip_on=False,
                    #     )
                    #     axes[idx].hlines(
                    #         xmin=cycle_len - delay - 1,
                    #         xmax=cycle_len - 1,
                    #         y=averages[len(dim) - delay - 1],
                    #         color=latest_plot[0].get_color(),
                    #         linewidth=0.5,
                    #         linestyle="--",
                    #         zorder=-1,
                    #         clip_on=False,
                    #     )
                    delay += 1
        axes[idx].legend(loc="upper right")

    markers = []
    for delay in range(delay):
        markers.append(
            matplotlib.lines.Line2D(
                [],
                [],
                color="blue",
                marker=f"{(delay % 4) + 1}",
                markersize=15,
                linestyle="",
                label=f"delay {delay}",
                alpha=0.5,
            )
        )

    axes[idx + 1].legend(
        handles=markers,
        # bbox_to_anchor=(0.0, 1.0, 1.0, 0.10),
        loc=3,
        ncol=2,
        mode="expand",
        borderaxespad=0.0,
    )

    # If there are any empty subplots, turn them off
    for ax in axes[num_nodes:]:
        ax.axis("off")

    # plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close("all")


def draw_individual_node_output(
    cycle_len,
    cycle_delay_max,
    network_input,
    all_outputs,
    filename,
    trials,
    *,
    include_previous=False,
    only_last=False,
    only_last_of_previous=False,
    custom_range=None,
    last_max_delay=False,
    end_tests=0,
    log_level=0,
    node_names={
        -1: "ready",
        -2: "go",
        0: "output",
        # 1: "output2",
    },
    average=False,
    delay_buckets=False,
    colors=standard_colors,
):
    max_foreperiod = cycle_len - cycle_delay_max
    only_last = only_last or last_max_delay
    dim = np.arange(-1 if only_last_of_previous else -cycle_len, cycle_len, 1)
    # dim = np.append(np.arange(1, cycle_len), np.arange(cycle_len))

    nodes = all_outputs[0][0].keys()

    # Calculate the number of rows needed for a 3-column layout
    num_nodes = len(nodes)

    num_rows = (
        num_nodes + 2 + 3
    ) // 3  # Add 2 to ensure that we have enough rows for all nodes

    #  Plot each node in its respective subplot
    previous_output = [None for _ in nodes]
    previous_fp = None
    this_fp = None

    delay_split = process_output(
        cycle_len,
        cycle_delay_max,
        all_outputs,
        trials,
        include_previous=include_previous,
        only_last=only_last,
        only_last_of_previous=only_last_of_previous,
        custom_range=custom_range,
        last_max_delay=last_max_delay,
        end_tests=end_tests,
        shape=(
            "foreperiod_blocks",
            "node_count",
            "delay_block",
            "trials",
            "max_trial_len",
        ),
    )
    for block in range(len(all_outputs)):
        # Create subplots in a 3xN grid
        fig, axes = plt.subplots(
            num_rows, 3, figsize=(15, num_rows * 5 + 1), sharey=True
        )  # Adjust the figsize as necessary

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        for idx, node in enumerate(nodes):
            if node == -2:
                this_fp = np.where(delay_split[block][idx] == 1)[0][0]
            # plt.xlim((dim[0], dim[-1]))
            axes[idx].set_title(
                node_names[node] if node in node_names else f"Node {node}"
            )
            axes[idx].set_xlabel("Timestep")
            axes[idx].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
            if idx % 3 == 0:
                axes[idx].set_ylabel("Magnitude")
            for x in [
                0 if only_last_of_previous else -cycle_delay_max,
                0,
                max_foreperiod,
            ]:
                axes[idx].axvline(
                    x=x,
                    ymin=0.0,
                    ymax=1,
                    c="gray",
                    linewidth=1,
                    linestyle="-" if x == 0 else "--",
                    zorder=-1,
                    clip_on=False,
                )

            entries = np.concatenate(
                [
                    (
                        [previous_output[idx]]
                        if np.any(previous_output[idx])
                        else np.full(delay_split[block][idx][0].shape, np.nan)
                    ),
                    *delay_split[block][idx],
                ]
            )
            if delay_buckets:
                delay_plots = []
                markers = []
                overall_std = np.nanstd(entries, axis=0)
                axes[idx].plot(
                    dim,
                    np.where(overall_std <= 0.02, overall_std, np.nan),
                    marker="o",
                    markersize=10,
                    linestyle="",
                    c="black",
                )
                for delay_index, delay_block in enumerate(delay_split[block][idx]):
                    delay_plots.append(
                        axes[idx].plot(
                            dim,
                            np.nanmean(delay_block, axis=0),
                            c=colors[delay_index],
                            alpha=0.7,
                        )
                    )
                    delay_std = np.nanstd(delay_block, axis=0)
                    markers.append(
                        axes[idx].plot(
                            dim,
                            np.where(delay_std <= 0.02, delay_std, np.nan),
                            marker=f"{(delay_index % 4) + 1}",
                            markersize=15,
                            linestyle="",
                            c=colors[delay_index],
                        )
                    )
                axes[idx].legend(
                    handles=[delay_plot[0] for delay_plot in markers],
                    labels=[i for i in range(len(delay_plots))],
                    loc="upper right",
                )
            else:
                for entry_index, entry in enumerate(entries):
                    axes[idx].plot(
                        dim,
                        entry,
                        label=(
                            entry_index
                            if previous_output[idx] == None or entry_index != 0
                            else f"fp{previous_fp}"
                        ),
                    )
                axes[idx].legend(loc="upper right")
                axes[idx].plot(
                    dim,
                    np.nanmean(delay_block[idx][block], axis=0),
                    c=colors[block],
                    label=block + 1,
                    alpha=0.7,
                )
                axes[idx].legend(loc="upper right")
            previous_output[idx] = entries[-1]
            previous_fp = this_fp

            # If there are any empty subplots, turn them off
            for ax in axes[num_nodes:]:
                ax.axis("off")

            fig.savefig(f"{filename}-fp{block+1}", dpi=300, bbox_inches="tight")
    plt.close("all")


def _draw_individual_node_output(
    cycle_len,
    cycle_delay_max,
    network_input,
    all_outputs,
    filename,
    trials,
    *,
    include_previous=False,
    only_last=False,
    only_last_of_previous=False,
    custom_range=None,
    last_max_delay=False,
    end_tests=0,
    log_level=0,
    node_names={
        -1: "ready",
        -2: "go",
        0: "output",
        # 1: "output2",
    },
    average=False,
    delay_buckets=False,
    colors=standard_colors,
):
    max_foreperiod = cycle_len - cycle_delay_max
    only_last = only_last or last_max_delay
    indices = np.empty((len(all_outputs)), dtype=object)
    dim = np.arange(-1 if only_last_of_previous else -cycle_len, cycle_len, 1)
    # dim = np.append(np.arange(1, cycle_len), np.arange(cycle_len))

    previous_fp = None

    delay_split = np.empty(
        (len(all_outputs), cycle_delay_max, trials, len(dim)), dtype=float
    )
    delay_split.fill(np.nan)

    nodes = all_outputs[0][0].keys()

    # Calculate the number of rows needed for a 3-column layout
    num_nodes = len(nodes)
    num_rows = (
        num_nodes + 2
    ) // 3  # Add 2 to ensure that we have enough rows for all nodes

    #  Plot each node in its respective subplot
    previous_output = [None for _ in nodes]
    for block in range(len(all_outputs)):
        # Create subplots in a 3xN grid
        fig, axes = plt.subplots(
            num_rows, 3, figsize=(15, num_rows * 5 + 1), sharey=True
        )  # Adjust the figsize as necessary

        # Flatten the axes array for easy iteration
        axes = axes.flatten()
        for idx, node in enumerate(nodes):
            # plt.xlim((dim[0], dim[-1]))
            axes[idx].set_title(
                node_names[node] if node in node_names else f"Node {node}"
            )
            axes[idx].set_xlabel("Timestep")
            axes[idx].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
            if idx % 3 == 0:
                axes[idx].set_ylabel("Magnitude")
            for x in [
                0 if only_last_of_previous else -cycle_delay_max,
                0,
                max_foreperiod,
            ]:
                axes[idx].axvline(
                    x=x,
                    ymin=0.0,
                    ymax=1,
                    c="gray",
                    linewidth=1,
                    linestyle="-" if x == 0 else "--",
                    zorder=-1,
                    clip_on=False,
                )
            node_outputs = np.array(
                [entry[node] for entry in all_outputs[block]], dtype=float
            )
            if node == -2:
                this_fp = np.where(node_outputs == 1)[0][0]
            if node == -1:
                indices[block] = np.append(
                    np.asarray(np.where(node_outputs == 1))[0], len(node_outputs)
                )
                if end_tests:
                    indices[block] = indices[block][0:-(end_tests)]
                if only_last:
                    if last_max_delay:
                        for index in range(-1, -len(indices) + 2, -1):
                            if (
                                indices[block][index - 2] - indices[block][index]
                                == cycle_len * 2 + 1
                            ):
                                indices[block] = indices[block][index - 2 : index]
                                break
                    if indices[block][-1] - indices[block][0] > cycle_len * 2:
                        indices[block] = indices[block][-3:]
                if custom_range:
                    indices[block] = indices[block][custom_range[0] : custom_range[1]]

            block_indices = indices[block]
            entries = previous_output[idx] if previous_output[idx] != None else []
            if only_last_of_previous:
                entries.append(
                    np.concatenate(
                        (
                            [0],
                            np.pad(
                                node_outputs[block_indices[0] : block_indices[1]],
                                (
                                    0,
                                    cycle_len
                                    - len(
                                        node_outputs[
                                            block_indices[0] : block_indices[1]
                                        ]
                                    ),
                                ),
                                constant_values=(np.nan,),
                            ),
                        )
                    )
                )
            # print(f"{previous_output=}")
            # print(f"{entries=}")
            for i in range(len(block_indices) - 2):
                # print(f"{block_indices[i]}:{block_indices[i+1]}")

                if only_last_of_previous:
                    first_half = node_outputs[block_indices[i]]
                else:
                    first_half = np.pad(
                        node_outputs[block_indices[i] : block_indices[i + 1]],
                        (
                            0,
                            cycle_len
                            - len(
                                node_outputs[block_indices[i] : block_indices[i + 1]]
                            ),
                        ),
                        constant_values=(np.nan,),
                    )
                second_half = np.pad(
                    node_outputs[block_indices[i + 1] : block_indices[i + 2]],
                    (
                        0,
                        cycle_len
                        - len(
                            node_outputs[block_indices[i + 1] : block_indices[i + 2]]
                        ),
                    ),
                    constant_values=(np.nan,),
                )
                # print(delay_split)
                delay_first = (
                    block_indices[i + 1] - block_indices[i] - (max_foreperiod + 1)
                )
                delay_split[block][delay_first][i] = np.append(first_half, second_half)
                entries.append(np.append(first_half, second_half))
            entries = np.array(entries)
            if delay_buckets:
                # print(delay_split[block].shape)
                delay_plots = []
                markers = []
                overall_std = np.nanstd(entries, axis=0)
                axes[idx].plot(
                    dim,
                    np.where(overall_std <= 0.02, overall_std, np.nan),
                    marker="o",
                    markersize=10,
                    linestyle="",
                    c="black",
                )
                for delay_index, delay_block in enumerate(delay_split[block]):
                    axes[idx].plot(
                        dim,
                        delay_block.T,
                        c=colors[delay_index],
                        linestyle="--",
                    )
                    delay_plots.append(
                        axes[idx].plot(
                            dim,
                            np.nanmean(delay_block, axis=0),
                            c=colors[delay_index],
                        )
                    )
                    # for entry in delay_block:
                    # print(delay_block[0])
                    delay_std = np.nanstd(delay_block, axis=0)
                    # print(delay_std)
                    markers.append(
                        axes[idx].plot(
                            dim,
                            np.where(delay_std <= 0.02, delay_std, np.nan),
                            marker=f"{(delay_index % 4) + 1}",
                            markersize=15,
                            linestyle="",
                            c=colors[delay_index],
                        )
                    )
                axes[idx].legend(
                    handles=[delay_plot[0] for delay_plot in markers],
                    labels=[i for i in range(len(delay_plots))],
                    loc="upper right",
                )
            else:
                for entry_index, entry in enumerate(entries):
                    axes[idx].plot(
                        dim,
                        entry,
                        label=(
                            entry_index
                            if previous_output[idx] == None or entry_index != 0
                            else f"fp{previous_fp}"
                        ),
                    )
                axes[idx].legend(loc="upper right")
            # If there are any empty subplots, turn them off
            for ax in axes[num_nodes:]:
                ax.axis("off")

            # print(f"{include_previous=}")
            if include_previous:
                previous_output[idx] = [np.append(first_half, second_half)]
                # print(f"{previous_output=}")

        # plt.tight_layout()
        fig.savefig(f"{filename}-fp{block+1}", dpi=300, bbox_inches="tight")
        previous_fp = this_fp

    plt.close("all")


def draw_foreperiod_adaptation(
    cycle_len,
    cycle_delay_max,
    network_input,
    all_outputs,
    filename,
    trials,
    *,
    include_previous=False,
    only_last=False,
    only_last_of_previous=False,
    custom_range=None,
    last_max_delay=False,
    end_tests=0,
    node_names={
        -1: "ready",
        -2: "go",
        0: "output",
        # 1: "output2",
    },
    average=False,
    delay_buckets=False,
    colors=standard_colors,
):
    max_foreperiod = cycle_len - cycle_delay_max
    only_last = only_last or last_max_delay
    dim = np.arange(-1 if only_last_of_previous else -cycle_len, cycle_len, 1)
    # dim = np.append(np.arange(1, cycle_len), np.arange(cycle_len))

    nodes = all_outputs[0][0].keys()

    # Calculate the number of rows needed for a 3-column layout
    num_nodes = len(nodes)

    num_rows = (
        num_nodes + 2 + 3
    ) // 3  # Add 2 to ensure that we have enough rows for all nodes

    delay_split = process_output(
        cycle_len,
        cycle_delay_max,
        all_outputs,
        trials,
        include_previous=include_previous,
        only_last=only_last,
        only_last_of_previous=only_last_of_previous,
        custom_range=custom_range,
        last_max_delay=last_max_delay,
        end_tests=end_tests,
    )

    for delay_index, delay_block in enumerate(delay_split):
        # Create subplots in a 3xN grid
        fig, axes = plt.subplots(
            num_rows, 3, figsize=(15, num_rows * 5 + 1), sharey=True
        )  # Adjust the figsize as necessary

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        for idx, node in enumerate(nodes):
            # plt.xlim((dim[0], dim[-1]))
            axes[idx].set_title(
                node_names[node] if node in node_names else f"Node {node}"
            )
            axes[idx].set_xlabel("Timestep")
            axes[idx].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
            if idx % 3 == 0:
                axes[idx].set_ylabel("Magnitude")
            for x in [
                0 if only_last_of_previous else -cycle_delay_max,
                0,
                max_foreperiod,
            ]:
                axes[idx].axvline(
                    x=x,
                    ymin=0.0,
                    ymax=1,
                    c="gray",
                    linewidth=1,
                    linestyle="-" if x == 0 else "--",
                    zorder=-1,
                    clip_on=False,
                )
            for block in range(len(all_outputs)):
                axes[idx].plot(
                    dim,
                    np.nanmean(delay_block[idx][block], axis=0),
                    c=colors[block],
                    label=block + 1,
                )
            axes[idx].legend(loc="upper right")
            # If there are any empty subplots, turn them off
            for ax in axes[num_nodes:]:
                ax.axis("off")

        fig.savefig(f"{filename}-delay{delay_index}", dpi=300, bbox_inches="tight")
    plt.close("all")


def draw_average_node_output_around_go(
    cycle_len,
    cycle_delay_max,
    network_input,
    all_outputs,
    filename,
    trials,
    only_last=False,
    last_max_delay=False,
    end_tests=0,
    log_level=0,
    node_names={
        -1: "ready",
        -2: "go",
        0: "output",
        # 1: "output2",
    },
):
    max_foreperiod = cycle_len - cycle_delay_max
    only_last = only_last or last_max_delay
    # ready_indices = np.empty((len(all_outputs)), dtype=object)
    indices = np.empty((len(all_outputs)), dtype=object)
    dim = np.arange(-cycle_len, cycle_len, 1)

    delay_split = np.empty(
        (cycle_delay_max, len(all_outputs), trials, len(dim)), dtype=float
    )
    delay_split.fill(np.nan)

    nodes = all_outputs[0][0].keys()

    # Calculate the number of rows needed for a 3-column layout
    num_nodes = len(nodes)
    num_rows = (
        num_nodes + 2 + 3
    ) // 3  # Add 2 to ensure that we have enough rows for all nodes

    # Create subplots in a 3xN grid
    fig, axes = plt.subplots(
        num_rows, 3, figsize=(15, num_rows * 5 + 1), sharey=True
    )  # Adjust the figsize as necessary

    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    delay = 0
    #  Plot each node in its respective subplot
    for idx, node in enumerate(nodes):
        axes[idx].set_title(node_names[node] if node in node_names else f"Node {node}")
        axes[idx].set_xlabel("Timestep")
        axes[idx].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        if idx % 3 == 0:
            axes[idx].set_ylabel("Magnitude")
        for x in [
            0,
        ]:
            axes[idx].axvline(
                x=x,
                ymin=0.0,
                ymax=1,
                c="gray",
                linewidth=1,
                linestyle="-" if x == 0 else "--",
                zorder=-1,
                clip_on=False,
            )
        for block in range(len(all_outputs)):
            node_outputs = np.array(
                [entry[node] for entry in all_outputs[block]], dtype=float
            )
            if node == -1:
                # print([entry[-2] for entry in all_outputs[block]])
                indices[block] = np.asarray(
                    np.where(
                        np.array(
                            [entry[-2] for entry in all_outputs[block]], dtype=float
                        )
                        == 1
                    )
                )[
                    0
                ]  # Split on go?
                # print(indices[block])
                # ready_indices[block] = np.append(
                #     np.asarray(np.where(node_outputs == 1))[0], len(node_outputs)
                # )
                if end_tests:
                    indices[block] = indices[block][0:-(end_tests)]
                    # ready_indices[block] = ready_indices[block][0:-(end_tests)]
                if only_last:
                    if last_max_delay:
                        for index in range(-1, -len(indices) + 2, -1):
                            if (
                                indices[block][index - 2] - indices[block][index]
                                == cycle_len * 2 + 1
                            ):
                                indices[block] = indices[block][index - 2 : index]
                                break
                        # for index in range(-1, -len(ready_indices) + 2, -1):
                        #     if (
                        #         ready_indices[block][index - 2]
                        #         - ready_indices[block][index]
                        #         == cycle_len * 2 + 1
                        #     ):
                        #         ready_indices[block] = ready_indices[block][
                        #             index - 2 : index
                        #         ]
                        #         break
                    if indices[block][-1] - indices[block][0] > cycle_len * 2:
                        indices[block] = indices[block][-3:]
                        # ready_indices[block] = ready_indices[block][-3:]
            block_indices = indices[block]
            # ready_block_indices = ready_indices[block]
            entries = []
            for i in range(len(block_indices) - 2):
                # print(f"{block_indices[i]}:{block_indices[i+1]}")
                # print(block_indices)
                delay_first = (
                    block_indices[i + 1]
                    - block_indices[i]
                    - (max_foreperiod + 1)
                    # + (len(indices) - block)
                )
                delay_second = (
                    block_indices[i + 2]
                    - block_indices[i + 1]
                    - (max_foreperiod + 1)
                    # + (len(indices) - block)
                )
                block_offset = max_foreperiod - block
                first_half = np.append(
                    np.pad(
                        node_outputs[
                            block_indices[i] : block_indices[i] + block_offset
                        ],
                        (
                            0,
                            cycle_len
                            - len(
                                node_outputs[block_indices[i] : block_indices[i + 1]]
                            ),
                        ),
                        constant_values=(np.nan,),
                    ),
                    node_outputs[
                        block_indices[i] + block_offset : block_indices[i + 1]
                    ],
                )
                second_half = np.append(
                    np.pad(
                        node_outputs[
                            block_indices[i + 1] : block_indices[i + 1] + block_offset
                        ],
                        (
                            0,
                            cycle_len
                            - len(
                                node_outputs[
                                    block_indices[i + 1] : block_indices[i + 2]
                                ]
                            ),
                        ),
                        constant_values=(np.nan,),
                    ),
                    node_outputs[
                        block_indices[i + 1] + block_offset : block_indices[i + 2]
                    ],
                )
                delay_split[delay_first][block][i] = np.pad(
                    np.append(first_half, second_half[0]),
                    (
                        0,
                        len(dim) - cycle_len - 1,
                    ),
                    constant_values=(np.nan,),
                )
                delay_split[delay_second][block][i] = np.pad(
                    second_half,
                    (len(dim) - cycle_len, 0),
                    constant_values=(np.nan,),
                )
                entries.append(np.append(first_half, second_half))
            entries = np.array(entries)
            # print(indices)
            # print(entries)
            latest_plot = axes[idx].plot(
                dim,
                entries[0] if only_last else np.nanmean(entries, axis=0),
                label=block + 1,
            )
            if not only_last:
                # print(
                #     node_names[node] if node in node_names else f"Node {node}",
                #     "block ",
                #     block,
                # )
                delay = 0
                for delayed_block in delay_split:
                    # print(delayed_block)
                    averages = np.nanmean(delayed_block[block], axis=0)
                    # averages[cycle_len - 1] = averages[max_foreperiod + delay]
                    # averages[cycle_len * 2 - 1] = averages[
                    #     cycle_len + max_foreperiod + delay
                    # ]
                    # mask = np.concatenate(
                    #     (
                    #         np.full(max_foreperiod, False),
                    #         np.isfinite(averages[max_foreperiod:cycle_len]),
                    #         np.full(max_foreperiod, True),
                    #         np.isfinite(averages[max_foreperiod + cycle_len :]),
                    #     ),
                    #     axis=None,
                    # )
                    # print(mask, "len", len(mask))
                    # print(dim[mask])
                    # print(latest_plot)
                    # print("delay: ", delay, averages)
                    # print(averages)
                    # print("delay index", -(cycle_delay_max - delay))
                    # print(cycle_len - (cycle_delay_max - delay))
                    axes[idx].plot(
                        dim,
                        averages,
                        marker=f"{(delay % 4) + 1}",
                        markersize=5.0,
                        linewidth=0.3,
                        # linestyle="",
                        color=latest_plot[0].get_color(),
                        alpha=0.7,
                    )
                    # if delay < len(delay_split) - 1:
                    #     axes[idx].hlines(
                    #         xmin=-(cycle_delay_max - delay),
                    #         xmax=-1,
                    #         y=averages[cycle_len - (cycle_delay_max - delay)],
                    #         color=latest_plot[0].get_color(),
                    #         linewidth=0.5,
                    #         linestyle="--",
                    #         zorder=-1,
                    #         clip_on=False,
                    #     )
                    #     axes[idx].hlines(
                    #         xmin=cycle_len - delay - 1,
                    #         xmax=cycle_len - 1,
                    #         y=averages[len(dim) - delay - 1],
                    #         color=latest_plot[0].get_color(),
                    #         linewidth=0.5,
                    #         linestyle="--",
                    #         zorder=-1,
                    #         clip_on=False,
                    #     )
                    delay += 1
        axes[idx].legend(loc="upper right")

    markers = []
    for delay in range(delay):
        markers.append(
            matplotlib.lines.Line2D(
                [],
                [],
                color="blue",
                marker=f"{(delay % 4) + 1}",
                markersize=15,
                linestyle="",
                label=f"delay {delay}",
                alpha=0.5,
            )
        )

    axes[idx + 1].legend(
        handles=markers,
        # bbox_to_anchor=(0.0, 1.0, 1.0, 0.10),
        loc=3,
        ncol=2,
        mode="expand",
        borderaxespad=0.0,
    )

    # If there are any empty subplots, turn them off
    for ax in axes[num_nodes:]:
        ax.axis("off")

    # plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close("all")


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
    log_level=0,
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
    plt.close("all")
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
        plt.close("all")
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
        if log_level > 2:
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
                plt.close("all")
            if (
                network
                and log_level > 3
                and test_names[test]
                in [
                    "trial_last",
                    # "trial_first",
                    "go_omitted",
                ]
            ):
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


def extract_key_and_fitness_from_file(file_path):
    """Extract the key and fitness value from a given file."""
    key = None
    fitness = None
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("Key:"):
                key = line.split(":")[1].strip()
            elif line.startswith("Fitness:"):
                fitness = float(line.split(":")[1].strip())
            if key is not None and fitness is not None:
                break
    return key, fitness


def scan_folders_and_extract_info(root_folder, target_file="genome.txt"):
    """Scan through folders starting from 'root_folder' and extract key and fitness values from files."""
    fitness_dict = {}
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file == target_file:
                file_path = os.path.join(root, file)
                key, fitness = extract_key_and_fitness_from_file(file_path)
                if key is not None and fitness is not None:
                    fitness_dict[key] = fitness
    return fitness_dict


def group_results_by_key_content(results, patterns):
    """Group results based on the content of the keys."""
    grouped_results = {pattern: {} for pattern in list(patterns.values())}
    for key, value in results.items():
        for pattern in list(patterns.keys()):
            if pattern in key:
                grouped_results[patterns[pattern]][key] = value
                break  # Assuming each key only belongs to one pattern group
    return grouped_results


def calculate_statistics(results):
    """Calculate average, max, and min fitness for each grouping."""
    stats = {}
    for key, fitness_values in results.items():
        if fitness_values:
            avg_fitness = sum(fitness_values.values()) / len(fitness_values)
            max_fitness = max(fitness_values.values())
            min_fitness = min(fitness_values.values())
            stats[key] = (avg_fitness, max_fitness, min_fitness)
    return stats


def plot_all_statistics(grouped_results, filename, colors=standard_colors):
    """Plot the statistics for all groupings in different subplots."""
    num_folders = len(grouped_results)
    fig = plt.figure()

    # plt.setp(axes, ylim=(0.85, 1))

    # if num_folders == 1:
    #     axes = [axes]  # Ensure axes is iterable even for a single subplot

    # labels = [
    #     entry.split("hidden_nodes")[1].split("-")[0].replace("_", " ").strip()
    #     for entry in list(list(grouped_results.values())[0].keys())
    # ]

    for i, (folder_name, results) in enumerate(grouped_results.items()):
        stats = calculate_statistics(results)
        labels = [
            entry.split("hidden_nodes")[1].split("-")[0].replace("_", " ").strip()
            for entry in list(stats.keys())
        ]
        labels[:] = ["normal" if x == "" else x for x in labels]
        avg_values = [val[0] for val in stats.values()]
        error_values = [
            (val[0] - val[2], val[1] - val[0]) for val in stats.values()
        ]  # (avg - min, max - avg)

        x = np.arange(len(labels))  # the label locations

        # Plotting the average values as dots
        # plt.scatter(x, avg_values, c=colors[i])

        # Adding error bars to represent the range between min and max
        plt.errorbar(
            x + 0.1 * i,
            avg_values,
            yerr=list(zip(*error_values)),
            fmt="o",
            color=colors[i],
            ecolor="lightgray",
            elinewidth=3,
            capsize=0,
            label=folder_name,
        )

    # ax.set_xlabel("Groups")
    plt.ylabel("Fitness")
    plt.title(f"Fitness Statistics")
    plt.xticks(x + 0.05 * i, labels)
    lgd = plt.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close("all")


def calculate_and_plot_statistics(root_folder, patterns, filename="fitness_statistics"):
    """Calculate and plot statistics for the given folders and patterns."""
    # Scan folders and extract info
    results = {}
    for root, folders, files in os.walk(root_folder):
        for folder in folders:
            folder_name = os.path.basename(folder)
            results[folder] = scan_folders_and_extract_info(os.path.join(root, folder))
    grouped_results = group_results_by_key_content(results, patterns)

    # Plot and calculate statistics
    plot_all_statistics(grouped_results, filename)


def plot_fitness_over_time(
    folder_list, csv_filename, output_filename="fitness_over_time", columns=3
):
    """Plot the best and average fitness over time for each population in the folder list."""
    num_rows = len(folder_list) // columns + 1
    fig, axes = plt.subplots(
        num_rows, columns, figsize=(20, num_rows * 5 + 1), sharey=True
    )
    plt.yscale("logit")

    if len(folder_list) == 1:
        axes = [axes]  # Ensure axes is iterable even for a single subplot

    # Flatten the axes array for easy iteration
    # axes = axes.flatten()

    for index, folder in enumerate(folder_list):
        csv_file_path = os.path.join(folder, csv_filename)
        if os.path.exists(csv_file_path):
            # Read the CSV file
            df = pd.read_csv(
                csv_file_path, header=None, names=["Best Fitness", "Average Fitness"]
            )

            y = index % columns
            x = index // columns

            best = df["Best Fitness"]
            average = df["Average Fitness"]
            # Plotting the best and average fitness
            axes[x][y].plot(best, label="Best Fitness", color="blue")
            axes[x][y].plot(average, label="Average Fitness", color="red")

            axes[x][y].set_xlabel("Time Step")
            axes[x][y].set_ylabel("Fitness")
            axes[x][y].set_title(
                os.path.basename(folder).split("no-reset")[1].split("-")[0]
            )
            axes[x][y].legend()

    plt.tight_layout()
    fig.savefig(output_filename, dpi=300)
    plt.close("all")


def plot_fitness_over_time_for_subfolders(
    root_folder,
    csv_filename="fitness_over_time.csv",
    output_filename="fitness_over_time",
):
    """Plot the best and average fitness over time for each population in the given root folder."""
    folder_list = [
        os.path.join(root_folder, folder)
        for folder in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, folder))
    ]
    plot_fitness_over_time(folder_list, csv_filename, output_filename)


def network_output_matrix(
    *,
    network_input,
    all_outputs,
    filename,
    foreperiods,
    cycle_len,
    cycle_delay_max,
    # network,
    include_previous=False,
    only_last=False,
    custom_range=None,
    last_max_delay=False,
    end_tests=0,
    log_level=0,
    node_names={
        -1: "ready",
        -2: "go",
        0: "output",
        # 1: "output2",
    },
):
    max_foreperiod = cycle_len - cycle_delay_max
    network = f'network{filename.split("network")[1].strip("/")}'
    df = pd.DataFrame(
        columns=[
            "network",
            "fp",
            "prev_delay",
            "trial",
            "fp_order",
            *all_outputs[0][0].keys(),
        ]
    )
    # indices = np.empty((len(all_outputs)), dtype=object)
    for idx, block in enumerate(all_outputs):
        fp = foreperiods[idx]
        prev_delay = 0
        trial = 0
        node_outputs = np.array([entry[-1] for entry in block], dtype=float)
        block_indices = np.append(
            np.asarray(np.where(node_outputs == 1))[0], len(node_outputs)
        )
        for entry in block:
            df.loc[len(df.index)] = [
                network,
                fp,
                prev_delay,
                trial,
                foreperiods,
                *entry.values(),
            ]
            prev_delay = (
                block_indices[trial + 1] - block_indices[trial] - (max_foreperiod + 1)
            )

    df.to_csv(f"{filename}.csv")
