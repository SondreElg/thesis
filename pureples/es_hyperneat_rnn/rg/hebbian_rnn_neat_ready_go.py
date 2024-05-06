"""
An experiment using a NEAT network to perform the ready-go task.
Fitness threshold set in config
- by default very high to show the high possible accuracy of this library.
"""

import copy
import sys
import os
import posixpath
import pickle
import math
import argparse
import neat
import neat.nn
import multiprocessing
import json
import numpy as np
import shutil
import pureples
from itertools import combinations
from ast import literal_eval
from neat.graphs import required_for_output, creates_cycle
from pureples.shared.visualize import (
    draw_net,
    make_genome_matrix,
    draw_output,
    draw_hebbian,
    draw_omission_trials,
    plot_hebbian_correlation_heatmap,
    draw_average_node_output,
    draw_individual_node_output,
    draw_average_node_output_around_go,
    calculate_and_plot_statistics,
    plot_fitness_over_time_for_subfolders,
    network_output_matrix,
    draw_foreperiod_adaptation,
    process_output,
)
from pureples.shared.ready_go import ready_go_list, foreperiod_rg_list
from pureples.shared.population_plus import Population
from pureples.shared.hebbian_rnn import HebbianRecurrentNetwork
from pureples.shared.hebbian_rnn_plus import HebbianRecurrentDecayingNetwork
from pureples.shared.distributions import bimodal
from pureples.shared.IZNodeGene_plus import IZNN, IZGenome

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--gens", default="1")
parser.add_argument("--base_folder", default="experiments/rg")
parser.add_argument("--target_folder", default=None)
parser.add_argument("--suffix", default="")
parser.add_argument("--load", default=None)
parser.add_argument(
    "--config", default="pureples/es_hyperneat_rnn/rg/config_neat_ready_go"
)
parser.add_argument(
    "--hebbian_type", default="positive", choices=["positive", "signed", "unsigned"]
)  # not yet implemented
parser.add_argument("--binary_weights", default="False")
parser.add_argument("--firing_threshold", default="0.20")
parser.add_argument("--hebbian_learning_rate", default="0.05")
parser.add_argument("--experiment", default="fp")  # not yet implemented
parser.add_argument("--trials", default="50")
parser.add_argument("--max_foreperiod", default="25")
parser.add_argument("--foreperiods", default="[1, 2, 3, 4, 5]")
parser.add_argument(
    "--verification_foreperiods", default="[1, 2, 3, 4, 5]"
)  # not yet implemented
parser.add_argument("--ordering", default="[]")
parser.add_argument("--reset", default="False")
parser.add_argument("--model", default="rnn", choices=["rnn", "iznn", "rnn_d"])
parser.add_argument("--overwrite", default="False")
parser.add_argument("--end_test", default="0")
parser.add_argument("--flip_pad_data", default="True")
parser.add_argument("--trial_delay_range", default="[0, 3]")
parser.add_argument("--log_level", default="0", choices=["-1", "0", "1", "2", "3", "4"])
parser.add_argument("--lesion", default="False")
args = parser.parse_args()

args.reset = bool(literal_eval(args.reset))
args.binary_weights = bool(literal_eval(args.binary_weights))

log_level = int(args.log_level)

foreperiod = int(args.max_foreperiod)
trials = int(args.trials)
time_block_size = 5
cycle_delay_range = json.loads(args.trial_delay_range)
cycle_len = math.floor(foreperiod / time_block_size)
# identity_func = lambda x: x
training_setup = {
    "function": foreperiod_rg_list,
    "params": [
        foreperiod,
        trials,
        time_block_size,
        cycle_delay_range,
        json.loads(args.foreperiods),
        bool(literal_eval(args.flip_pad_data)),
        json.loads(args.ordering),
    ],
}

# Config for network
CONFIG = None
if args.model == "rnn" or args.model == "rnn_d":
    CONFIG = neat.config.Config(
        pureples.shared.genome_plus.DefaultGenome,
        neat.reproduction.DefaultReproduction,
        neat.species.DefaultSpeciesSet,
        neat.stagnation.DefaultStagnation,
        args.config,
    )
elif args.model == "iznn":
    CONFIG = neat.config.Config(
        IZGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config,
    )


def run_rnn(
    genome,
    net,
    ready_go_data,
    visualize="",
    cycle_len=0,
    key=None,
    end_tests=0,
    log_level=0,
):
    block = 0
    blocks = len(ready_go_data)

    network_fitness = np.empty((blocks))
    omission_trial_outputs = np.empty((blocks, cycle_len), dtype=object)
    foreperiods = np.empty((blocks), dtype=int)
    all_outputs = np.empty((blocks), dtype=object)

    blocks_of_interest = (
        1 + blocks // 2 if bool(literal_eval(args.flip_pad_data)) else blocks
    )
    for inputs, expected_output in ready_go_data:
        trial = 0
        timesteps = len(inputs)
        outputs = np.empty(timesteps)
        all_block_outputs = np.empty(timesteps, dict)
        fitness = np.empty(timesteps)
        fitness.fill(np.nan)
        last_fitness = 0.0
        steady = False
        network = net
        if args.reset:
            net.reset()
        for index, input in enumerate(inputs):
            ready = int(input == 1)
            go = int(input == 2)
            steady = (steady or ready) and not go

            if ready:
                trial += 1
            training = trials - trial >= 0
            if training == 0:
                network = copy.deepcopy(net)

            output = network.activate([ready, go], training)

            last_fitness = 1 - abs(output[0] - expected_output[index]) ** 2
            outputs[index] = output[0]
            all_block_outputs[index] = copy.deepcopy(network.ovalues)
            if trial >= 6 and training:
                fitness[index] = last_fitness
        network_fitness[block] = np.nanmean(fitness)
        # print(f"{key=} - {network_fitness=}")
        all_outputs[block] = all_block_outputs
        foreperiod = np.where(np.array(inputs) == 2)[0][0]
        foreperiods[block] = foreperiod
        if visualize and block < blocks_of_interest and log_level > 1:
            if end_tests:
                indices = np.asarray(np.where(np.array(inputs) == 1))[0]
                omission_trial_outputs[block] = outputs[
                    indices[-end_tests] : indices[-end_tests + 1]
                ]
            draw_hebbian(
                net,
                f"{visualize}/block{block+1}_fp{foreperiod}_hebbian.png",
                node_names={
                    -1: "ready",
                    -2: "go",
                    0: "output",
                },
            )
            if log_level > 2:
                draw_output(
                    cycle_len,
                    np.array(inputs),
                    np.array(outputs),
                    np.array(expected_output),
                    f"{visualize}/block{block+1}_fp{foreperiod}_outputs.png",
                    end_tests=end_tests,
                    all_outputs=all_block_outputs if log_level > 3 else [],
                    network={
                        "config": CONFIG,
                        "genome": genome,
                        "hebbians": net.hebbian_update_log,
                    },
                    draw_std=CONFIG.learning_rate > 0,
                )
        block += 1
    if visualize and end_tests and log_level > 0:
        draw_omission_trials(
            omission_trial_outputs[np.argsort(foreperiods[:blocks_of_interest])],
            f"{visualize}/omission_trials.png",
        )
        if CONFIG.learning_rate > 0:
            plot_hebbian_correlation_heatmap(
                posixpath.join(
                    visualize,
                    f"block{blocks_of_interest}_fp{foreperiods[blocks_of_interest-1]}_hebbian.csv",
                ),
                foreperiods,
                posixpath.join(visualize, "hebbian_correlation_heatmap.png"),
            )
        draw_individual_node_output(
            cycle_len,
            cycle_delay_range[1],
            np.array(inputs),
            all_outputs[np.argsort(foreperiods[:blocks_of_interest])],
            posixpath.join(visualize, "individual_node_outputs"),
            trials,
            include_previous=True,
            only_last_of_previous=True,
            delay_buckets=True,
            custom_range=[0, 7],
            end_tests=end_tests,
        )
        draw_average_node_output(
            cycle_len,
            cycle_delay_range[1],
            np.array(inputs),
            all_outputs[np.argsort(foreperiods[:blocks_of_interest])],
            posixpath.join(visualize, "node_output_average"),
            trials,
            end_tests=end_tests,
        )
        draw_average_node_output_around_go(
            cycle_len,
            cycle_delay_range[1],
            np.array(inputs),
            all_outputs[np.argsort(foreperiods[:blocks_of_interest])],
            posixpath.join(visualize, "node_output_average_around_go"),
            trials,
            end_tests=end_tests,
        )
        # draw_average_node_output(
        #     cycle_len,
        #     cycle_delay_range[1],
        #     np.array(inputs),
        #     all_outputs[np.argsort(foreperiods[:blocks_of_interest])],
        #     posixpath.join(visualize, "node_output_last"),
        #     trials,
        #     only_last=True,
        #     end_tests=end_tests,
        # )
        network_output_matrix(
            network_input=np.array(inputs),
            all_outputs=all_outputs[np.argsort(foreperiods[:blocks_of_interest])],
            filename=posixpath.join(visualize, "network_output_matrix"),
            foreperiods=foreperiods[:blocks_of_interest],
            cycle_len=cycle_len,
            cycle_delay_max=cycle_delay_range[1],
        )
        draw_foreperiod_adaptation(
            cycle_len,
            cycle_delay_range[1],
            np.array(inputs),
            all_outputs[np.argsort(foreperiods[:blocks_of_interest])],
            posixpath.join(visualize, "foreperiod_output_growth"),
            trials,
            include_previous=True,
            only_last_of_previous=True,
            delay_buckets=True,
            end_tests=end_tests,
        )
        process_output(
            cycle_len,
            cycle_delay_range[1],
            all_outputs,
            trials,
            end_tests=end_tests,
            filename=posixpath.join(visualize, "processed_output"),
        )
        for index, entry in enumerate(["first", "second", "third", "fourth"]):
            draw_average_node_output(
                cycle_len,
                cycle_delay_range[1],
                np.array(inputs),
                all_outputs[np.argsort(foreperiods[:blocks_of_interest])],
                posixpath.join(visualize, f"node_outputs_{entry}"),
                trials,
                custom_range=[index, index + 3],
                end_tests=end_tests,
            )
    if log_level == -1:
        draw_foreperiod_adaptation(
            cycle_len,
            cycle_delay_range[1],
            np.array(inputs),
            all_outputs[np.argsort(foreperiods[:blocks_of_interest])],
            posixpath.join(visualize, "foreperiod_output_growth"),
            trials,
            include_previous=True,
            only_last_of_previous=True,
            delay_buckets=True,
            end_tests=end_tests,
        )
        process_output(
            cycle_len,
            cycle_delay_range[1],
            all_outputs,
            trials,
            end_tests=end_tests,
            filename=posixpath.join(visualize, "processed_output"),
        )
    # print(f"{key=} - {np.mean(network_fitness)=}")
    return np.mean(network_fitness)


def _eval_fitness(genome, config):
    """
    Fitness function.
    Evaluate the fitness of a single genome.
    """

    if args.model == "rnn":
        network = HebbianRecurrentNetwork.create(genome, config)

        # genome fitness
        return run_rnn(None, network, config.train_set)
    elif args.model == "rnn_d":
        network = HebbianRecurrentDecayingNetwork.create(genome, config)

        # genome fitness
        return run_rnn(None, network, config.train_set)
    # elif args.model == "iznn":
    #     network = IZNN.create(genome, config)

    #     # genome fitness
    #     return run_iznn(None, network, config.train_set)


def eval_fitness(genomes, config):
    """
    Fitness function.
    For each genome evaluate its fitness.
    """
    # If I want multiple trials per network per generation, I need to find another solution
    setattr(config, "train_set", training_setup["function"](*training_setup["params"]))
    for _, genome in genomes:
        genome.fitness = _eval_fitness(genome, config)


def ini_pop(state, config, stats):
    pop = Population(config, state)
    add_pop_reporter(pop, stats)
    return pop


def add_pop_reporter(pop, stats):
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True))


def run(*, gens, max_trials=1, initial_pop=None):
    """
    Create the population and run the ready_go task by providing eval_fitness as the fitness function.
    Returns the winning genome and the statistics of the run.
    """

    setattr(CONFIG, "training_setup", training_setup)

    distributions = len(training_setup["params"][4])

    # Create population and train the network. Return winner of network running 100 episodes.
    print("First run")
    setattr(CONFIG, "trials", 1)
    stats_one = neat.StatisticsReporter()
    pop = (
        ini_pop(initial_pop, CONFIG, stats_one)
        if type(initial_pop) == list or initial_pop is None
        else initial_pop
    )
    pe = neat.ParallelEvaluator(12, _eval_fitness)
    species_one = pop.run(pe.evaluate, gens)

    all_time_best = pop.reporters.reporters[0].best_genome()
    pop.reporters.reporters[0].save_genome_fitness(
        filename=f"{save_dir}/pop_fitness_history.csv",
        delimiter=",",
    )
    pop.reporters.reporters[0].save_species_count(
        filename=f"{save_dir}/pop_speciation_history.csv",
        delimiter=",",
    )
    pop.reporters.reporters[0].save_species_fitness(
        filename=f"{save_dir}/pop_species_fitness_history.csv",
        delimiter=",",
        null_value=-1,
    )
    # Save population if wished reused and draw it to file.
    with open(
        f"{save_dir}/population.pkl",
        "wb",
    ) as output:
        pickle.dump(
            pop,
            output,
            pickle.HIGHEST_PROTOCOL,
        )

    return species_one, (stats_one), all_time_best  # , stats_ten, stats_hundred)


def extract_winning_species(species_set, winner, limit=10):
    species_winners = {k: None for k in species_set.keys()}
    species_winners[0] = winner
    for key, species in species_set.items():
        for genome in species.members.values():
            if (
                not species_winners[key]
                and genome.fitness
                or (
                    genome.fitness
                    and species_winners[key].fitness
                    and genome.fitness > species_winners[key].fitness
                )
            ):
                species_winners[key] = genome
    return dict(
        sorted(
            species_winners.items(),
            key=lambda x: x[1].fitness if x[1] else -1,
            reverse=True,
        )[:limit]
    )


# If run as script.
if __name__ == "__main__":
    setattr(CONFIG, "binary_weights", bool(args.binary_weights))
    setattr(CONFIG, "firing_threshold", float(args.firing_threshold))
    setattr(CONFIG, "learning_rate", float(args.hebbian_learning_rate))
    max_trial_len = cycle_len + cycle_delay_range[1]

    hebbian_magnitude = (
        "dynamic"
        if args.model != "iznn" and CONFIG.genome_config.response_mutate_rate
        else "static"
    ) + "_magnitude"
    node_weights = "binary" if args.binary_weights else "real"
    reset = "reset" if args.reset else "no-reset"
    folder_name = (
        args.target_folder
        if args.target_folder
        else f"{args.experiment}-{node_weights}-{args.model}-{args.hebbian_type}-{hebbian_magnitude}-{('%.2f' % float(args.firing_threshold)).split('.')[1]}-threshold-{reset}{args.suffix}"
    )

    if "c:" in folder_name.lower():
        save_dir = folder_name
    else:
        save_dir = posixpath.join(args.base_folder, folder_name)
    similar_run = 0
    if os.path.exists(save_dir) and bool(literal_eval(args.overwrite)):
        with os.scandir(save_dir) as entries:
            for entry in entries:
                if entry.is_file():
                    os.unlink(entry.path)
                else:
                    shutil.rmtree(entry.path)
    else:
        while os.path.exists(save_dir):
            similar_run += 1
            save_dir = save_dir.split("__")[0] + "__" + str(similar_run)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    processed_output_path = posixpath.join(save_dir, "processed_output/")
    if not os.path.exists(processed_output_path):
        os.mkdir(processed_output_path)

    shutil.copyfile(
        args.config,
        f"{save_dir}/{args.config.split('/')[-1]}",
    )
    with open(
        f"{save_dir}/args.txt",
        "w",
    ) as output:
        output.write(str(args))

    initial_pop = None
    if args.load:
        with open(args.load, "rb") as f:
            filename = args.load.split("/")[-1]
            if "population" in filename:
                initial_pop = pickle.load(f)
            else:
                WINNERS = {"winner": pickle.load(f)}
    if not args.load or initial_pop:
        result = run(
            gens=int(args.gens),
            initial_pop=initial_pop,
        )
        winner = result[2]  # All-time best
        print("\nBest genome:\n{!s}".format(winner))

        WINNERS = extract_winning_species(result[0][2], winner)

    test_set = (
        result[0][1]
        if not args.load
        else training_setup["function"](*training_setup["params"])
    )

    test_set_expanded = copy.deepcopy(test_set)
    end_tests = int(args.end_test)
    if end_tests:
        end_tests += 1
        for j in range(len(test_set)):
            foreperiod = np.where(test_set_expanded[j][0] == 2)[0][0]
            for i in [0, foreperiod]:
                extension = np.zeros((2, max_trial_len), dtype=int)
                extension[:, i] = [2, 1]
                extension[:, 0] = [1, 0]
                test_set_expanded[j][0] = np.append(
                    test_set_expanded[j][0], extension[0], axis=0
                )
                test_set_expanded[j][1] = np.append(
                    test_set_expanded[j][1], extension[1], axis=0
                )

    count = 0
    for WINNER in WINNERS.values():
        lesions = {}
        if bool(literal_eval(args.lesion)):
            WINNER = WINNER.get_pruned_copy(CONFIG.genome_config)
            lesion_list = []
            print(WINNER.connections)
            print("###########################")
            for cg in WINNER.connections.values():
                input_key, output_key = cg.key
                if input_key >= 0 and output_key > 0 and input_key != output_key:
                    lesion_list.append(cg.key)

            combined_list = []
            for r in range(1, len(lesion_list) + 1):
                # print(f"{combinations(lesion_list, r)=}")
                combined_list.extend(combinations(lesion_list, r))
            # print(lesion_list)
            # print(combined_list)
            combined_list = [list(comb) for comb in combined_list]
            # print(combined_list)
            for entry in combined_list:
                target = True
                lesion = copy.deepcopy(WINNER)
                for connection in entry:
                    required = required_for_output(
                        CONFIG.genome_config.input_keys,
                        CONFIG.genome_config.output_keys,
                        lesion.connections,
                    )
                    lesion.connections[connection].enabled = False
                    if not creates_cycle(lesion.connections, connection):
                        target = False
                        print(f"{connection} DOESN'T CREATE CYCLE")
                        break
                if target:
                    lesions[tuple(entry)] = lesion

        count += 1
        network_dir = posixpath.join(save_dir, f"network{count}/")
        if not os.path.exists(network_dir):
            os.mkdir(network_dir)
        lesions["(0)"] = WINNER
        lesion_count = 0
        for lesion, genome in lesions.items():
            # print(lesion)
            lesion_dir = posixpath.join(network_dir, f"{str(lesion)}/")
            if not os.path.exists(lesion_dir):
                os.mkdir(lesion_dir)
            # Verify network output against training data.
            print("\nOutput:")
            NETWORK = (
                HebbianRecurrentNetwork.create(genome, CONFIG)
                if args.model == "rnn"
                else HebbianRecurrentDecayingNetwork.create(genome, CONFIG)
            )
            NETWORK.reset()
            genome.fitness = run_rnn(
                genome,
                NETWORK,
                test_set_expanded,
                visualize=lesion_dir if log_level != 0 else "",
                cycle_len=max_trial_len,  # Assume cycle_len is same/larger for last dist
                key=lesion,
                end_tests=end_tests,
                log_level=log_level,
            )
            draw_net(
                CONFIG,
                genome,
                filename=f"{lesion_dir}network",
                node_names={
                    -1: "ready",
                    -2: "go",
                    -3: "fitness",
                    0: "output",
                    1: "output2",
                },
                node_colors={
                    -1: "yellow",
                    -2: "green",
                    -3: "grey",
                    0: "lightblue",
                    1: "lightblue",
                },
                detailed=True,
            )
            # Save network if wished reused and draw it to file.
            with open(
                f"{lesion_dir}genome.pkl",
                "wb",
            ) as output:
                pickle.dump(genome, output, pickle.HIGHEST_PROTOCOL)

            with open(
                f"{lesion_dir}network.pkl",
                "wb",
            ) as output:
                pickle.dump(NETWORK, output, pickle.HIGHEST_PROTOCOL)

            with open(
                f"{lesion_dir}genome.txt",
                "w",
            ) as output:
                output.write(str(genome))
            lesion_count += 1

        make_genome_matrix(
            CONFIG,
            WINNER,
            filename=f"{network_dir}network_matrix",
            prune_unused=True,
        )
    # calculate_and_plot_statistics(
    #     "c:/Users/Sondr/pureples/pureples/experiments/ready_go/meetings/01-30",
    #     {
    #         "1_hidden_nodes": "1_hidden_nodes",
    #         "2_hidden_nodes": "2_hidden_nodes",
    #         "3_hidden_nodes": "3_hidden_nodes",
    #         "4_hidden_nodes": "4_hidden_nodes",
    #     },
    #     f"{}/fitness_statistics",
    # )

    # plot_fitness_over_time_for_subfolders(
    #     "c:/Users/Sondr/pureples/pureples/experiments/ready_go/results/",
    #     "pop_fitness_history.csv",
    #     f"{}/fitness_over_time",
    # )

# TODO
# Save fitness history of each block?
## Can give insight into how volatile the changes caused by the Hebbian weight is

# Experiments
## Ideas
### Implement the experiment from the Maes et al. 2020 paper
### Prior vs posterior probability (12345 vs regular)
##!!! Test entire population on foreperiod outside learned foreperiod !!!
### Keep best, visualize
## Holdover at 80%

# Visualize
## Activity of each neuron near ready-signal for multiple slices of each block
### Seperated by delay - Ignored due to low difference between delay averages
### For only trials where the foreperiod remainder and trial delay add up to the same number (so that they're synced on both Go and Ready)
## Activity of each neuron near go-signal for multiple slices of each block
### For each cycle delay

# Next meeting
## PP of different types of neurons (decay, modulation, etc.)
### Slides of networks with 2 hidden nodes, their outputs and their matrices
## 3 nodes vs 4 nodes for larger datasets
