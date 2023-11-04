"""
Varying visualisation tools.
"""

import os
import pickle
import warnings
import graphviz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
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
    fmt="svg",
    detailed=False,
    node_outputs=None,
    hebbians=None,
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
            name = (
                f"{name}\nbias {'%.3f' % node.bias}\nh_scaling {'%.3f' % node.response}"
            )
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
            name = f"key {name}\nbias {'%.3f' % node.bias}\nh_scaling {'%.3f' % node.response}"
            if node_outputs:
                name = name + f"\noutput {'%.3f' % node_outputs[n]}"
            node_names[n] = name
        attrs = {"style": "filled", "fillcolor": node_colors.get(n, "white")}
        dot.node(name, _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = "solid" if cg.enabled else "dotted"
            color = "red" if cg.weight > 0 else "blue"
            width = str(0.1 + abs(cg.weight * 3))
            if hebbians:
                label = "{:.3f}".format(
                    hebbians.get(output, {}).get(input, 0)
                    * genome.nodes[output].response
                    + cg.weight
                )
            else:
                label = ""
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


def draw_hebbian(
    hebbian,
    filename,
):
    fig = plt.figure()
    df = pd.DataFrame(
        [
            {f"{k2}-{k}": v2 for d in i for k, v in d.items() for k2, v2 in v.items()}
            for i in hebbian
        ]
    )
    df.to_csv(filename.split(".")[0] + ".csv")
    plt.plot(df)
    lgd = plt.legend(df.columns, bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlabel("trial")
    plt.ylabel("magnitude")
    # plt.show()
    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_extra_artists=(lgd,), bbox_inches="tight")
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

    split_expected_output = np.array(
        np.split(expected_output, indices)[:-end_tests], dtype=object
    )
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
    last = split_output[len(indices) - end_tests]
    # max_out = np.max(split_output[30:-end_tests], axis=0)
    plt.plot(expected_avg, label="expected")
    plt.plot(avg, label="average")
    plt.plot(first, label="first")
    plt.plot(trained, label="trained")
    plt.plot(last, label="last")
    # plt.plot(max_out, label="maximum")
    lgd = plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.xlabel("timestep")
    plt.ylabel("magnitude")
    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close()
    if end_tests:
        end_test_set = split_output[-end_tests:]
        fig2 = plt.figure()
        plt.plot(expected_avg, label="expected_avg")
        # plt.plot(end_test_set[0], label="first")
        # plt.plot(end_test_set[1], label="middle")
        # plt.plot(end_test_set[2], label="end")
        plt.plot(end_test_set[3], label="omission")
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
    if len(all_outputs):
        df = pd.json_normalize(all_outputs)
        df.to_csv(filename.split(".")[0] + f"_all.csv")
        test_names = [
            "trial_last",
            "go_on_first",
            "go_on_middle",
            "go_on_end",
            "go_omitted",
            "trial_first",
        ]
        for test in range(end_tests + 2):
            df = pd.json_normalize(
                all_outputs[
                    indices[-(end_tests - test) - 1] : indices[-(end_tests - test)]
                    if end_tests - test
                    else None
                ]
            )
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
                "trial_first",
                # "go_omission",
            ]:
                try:
                    save_dir = (
                        filename.split(".")[0]
                        + f"_all_{test_names[test]}_detailed_network/"
                    )
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    hebbian_trial_index = (
                        len(network["hebbians"]) - len(indices + 1)
                        if test_names[test] == "trial_first"
                        else -1
                    )
                    for timestep in range(len(df.iloc[:, :])):
                        draw_net(
                            network["config"],
                            network["genome"],
                            filename=save_dir + f"timestep_{timestep}",
                            hebbians=network["hebbians"][hebbian_trial_index][0],
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
                except Exception as error:
                    print(f"Error: {error}")
                    print(
                        f"########\nDrawing detailed network of {filename} failed\n########\n{df=}"
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
