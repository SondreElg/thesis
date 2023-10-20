"""
Varying visualisation tools.
"""

import pickle
import graphviz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math


def draw_net(net, filename=None, node_names={}, node_colors={}):
    """
    Draw neural network with arbitrary topology.
    """
    node_attrs = {"shape": "circle", "fontsize": "9", "height": "0.2", "width": "0.2"}

    dot = graphviz.Digraph("svg", node_attr=node_attrs)

    inputs = set()
    for k in net.input_nodes:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {
            "style": "filled",
            "shape": "box",
            "fillcolor": node_colors.get(k, "lightgray"),
        }
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in net.output_nodes:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {"style": "filled", "fillcolor": node_colors.get(k, "lightblue")}
        dot.node(name, _attributes=node_attrs)

    # print(f"{net.node_evals=}")
    for entry in net.node_evals:
        node = entry[0]
        links = entry[5]
        for i, w in links:
            node_input, output = node, i
            a = node_names.get(output, str(output))
            b = node_names.get(node_input, str(node_input))
            style = "solid"
            color = "red" if w > 0.0 else "blue"
            width = str(0.1 + abs(w / 5.0))
            dot.edge(
                a, b, _attributes={"style": style, "color": color, "penwidth": width}
            )

    dot.render(filename)

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
    fig.savefig(filename, dpi=300)


def draw_hebbian(
    hebbian,
    filename,
):
    fig = plt.figure()
    # print(hebbian)
    for i in range(len(hebbian[0])):
        # print(hebbian[0][i])
        # print(hebbian[0][i].keys())
        ids = list(hebbian[0][i].keys())
        for id in ids:
            plt.plot([entry[i][id] for entry in hebbian], label=f"{i}-{id}")
    plt.legend()
    plt.xlabel("trial")
    plt.ylabel("magnitude")
    # plt.show()
    fig.savefig(filename, dpi=300)
    plt.close()


def draw_hist(
    cycle_len,
    network_input,
    network_output,
    expected_output,
    filename,
):
    fig = plt.figure()
    indices = np.asarray(np.where(network_input == 1))[0][1:]
    # print(f"{indices=}")

    split_output = np.array(np.split(network_output, indices), dtype=object)

    # print(f"{np.asarray(split_output)=}")

    split_expected_output = np.array(
        np.split(expected_output, indices)[:-4], dtype=object
    )
    split_expected_end_test_output = np.array(
        np.split(expected_output, indices)[-4:], dtype=object
    )
    # print(f"{split_expected_output=}")
    # print(f"{split_expected_end_test_output=}")

    # print(
    #     f"{len(split_output)=} | {len(network_output)=} | {len(split_expected_output)=} | {len(expected_output)=} | {len(indices)=}"
    # )

    # Pad arrays
    split_output = np.array(
        [np.pad(i, ((0, cycle_len - len(i)))) for i in split_output]
    )
    split_expected_output = np.array(
        [np.pad(i, ((0, cycle_len - len(i)))) for i in split_expected_output]
    )

    # Calculate average for each timestep
    expected_avg = np.average(split_expected_output[30:-4], axis=0)
    avg = np.average(split_output[30:-4], axis=0)
    max_out = np.max(split_output[30:-4], axis=0)
    # print(f"{expected_avg=}")
    # print(f"{avg=}")
    plt.plot(expected_avg, label="expected")
    plt.plot(avg, label="actual")
    plt.plot(max_out, label="maximum")
    plt.legend()
    plt.xlabel("timestep")
    plt.ylabel("magnitude")
    # plt.show()
    # for i, v in enumerate(network_output):
    #     if v == 1:
    #         temp.append(network_output.split())
    # count, bins, ignored = plt.hist(network_output, 30, density=True)
    # plt.plot(
    #     bins,
    #     1
    #     / (sigma * np.sqrt(2 * np.pi))
    #     * np.exp(-((bins - mu) ** 2) / (2 * sigma**2)),
    #     linewidth=2,
    #     color="r",
    # )
    fig.savefig(filename, dpi=300)
    plt.close()
    end_test_set = split_output[-4:]
    fig2 = plt.figure()
    # print(f"{split_output[:-4]=}")
    # print(f"{end_test_set=}")
    plt.plot(expected_avg, label="expected_avg")
    plt.plot(end_test_set[0], label="first")
    plt.plot(end_test_set[1], label="middle")
    plt.plot(end_test_set[2], label="end")
    plt.plot(end_test_set[3], label="none")
    plt.legend()
    plt.xlabel("timestep")
    plt.ylabel("output magnitude")

    # combined = np.append([expected_avg], end_test_set, 0)
    # print(combined)
    # unlabeled = plt.plot(combined)
    # plt.legend(unlabeled, ("expected avg", "first", "middle", "end", "none"))
    # plt.show()
    fig2.savefig(filename.split(".")[0] + "_end_tests.png", dpi=300)
    plt.close()
    return


def draw_tests(
    cycle_len,
    network_input,
    network_output,
    expected_output,
    filename,
):
    return
