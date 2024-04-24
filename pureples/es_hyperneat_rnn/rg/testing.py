from pureples.shared.visualize import (
    draw_net,
    make_genome_matrix,
    draw_pattern,
    draw_es,
    draw_hebbian,
    draw_average_node_output,
    draw_average_node_output_around_go,
    calculate_and_plot_statistics,
    plot_fitness_over_time_for_subfolders,
)

from subprocess import run
import os
import re

# TODO: Make visualization possible without rerunning populations/networks


def extract_args_from_string(filename):
    """Extract command line arguments from a .txt file and save them as key-value pairs in a dictionary,
    keeping the values as strings."""
    with open(filename) as args_file:
        args = args_file.readline()
        args_dict = {}
        # Extract the arguments substring
        args_str = re.findall(r"\((.*?)\)", args)[0]

        # Regular expression to match key-value pairs
        pattern = re.compile(r"(\w+)=('[^']*'|\"[^\"]*\"|\[[^\]]*\]|[^,]*)")
        matches = pattern.findall(args_str)

        for key, value in matches:
            # Strip quotes from the value, but keep it as a string
            value = value.strip("'\"")
            args_dict[key] = value
        return args_dict


config = None
population = None
args = None
root = "c:/Users/Sondr/pureples/experiments/rg/temper"
target_base = "c:/Users/Sondr/pureples/experiments/rg/meetings/04-24"
target_folder = None

dir_entries = os.listdir(root)
for folder in [
    folders for folders in dir_entries if os.path.isdir(os.path.join(root, folders))
]:
    folder_name = os.path.basename(folder)
    target_folder = os.path.join(target_base, folder_name).replace("\\", "/")
    for file in os.listdir(os.path.join(root, folder)):
        if "config" in file:
            config = os.path.join(root, folder, file).replace("\\", "/")
        if "population.pkl" in file:
            population = os.path.join(root, folder, file).replace("\\", "/")
        if "args.txt" in file:
            arguments = os.path.join(root, folder, file).replace("\\", "/")
    args = extract_args_from_string(arguments)
    args["gens"] = 0
    args["log_level"] = -1
    args["config"] = config
    args["load"] = population
    args["target_folder"] = target_folder
    args["lesion"] = 1
    args["flip_pad_data"] = 0
    args["ordering"] = [2, 3, 1, 0, 4]
    args["foreperiods"] = (
        [1, 2, 3, 4, 5, 6, 7, 8] if "more_fps" in folder_name else [1, 2, 3, 4, 5]
    )
    args["max_foreperiod"] = len(args["foreperiods"] * 5)
    args["ordering"] = [x for x in range(len(args["foreperiods"]))]

    args_string = " ".join([f'--{key} "{val}"' for key, val in args.items()])
    print(
        f"python pureples/es_hyperneat_rnn/rg/hebbian_rnn_neat_ready_go.py {args_string}\n"
    )
    run(
        f"python pureples/es_hyperneat_rnn/rg/hebbian_rnn_neat_ready_go.py {args_string}"
    )

calculate_and_plot_statistics(
    target_base,
    {
        "1_hidden_nodes": "1_hidden_nodes",
        "2_hidden_nodes": "2_hidden_nodes",
        "3_hidden_nodes": "3_hidden_nodes",
        "4_hidden_nodes": "4_hidden_nodes",
    },
    os.path.join(target_base, "fitness_statistics").replace("\\", "/"),
)

plot_fitness_over_time_for_subfolders(
    root,
    "pop_fitness_history.csv",
    # target_base,
    os.path.join(target_base, "fitness_over_time").replace("\\", "/"),
)

# calculate_and_plot_statistics(
#     "c:/Users/Sondr/pureples/pureples/experiments/ready_go/meetings/01-30",
#     {
#         "nodes-": "normal",
#         "long_delay-": "long_delay",
#         "long_delay_more_fps-": "long_delay_more_foreperiods",
#     },
#     "c:/Users/Sondr/pureples/pureples/experiments/ready_go/meetings/01-30/fitness_statistics",
# )
