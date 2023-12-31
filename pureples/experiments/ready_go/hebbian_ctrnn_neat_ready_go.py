"""
An experiment using a NEAT network to perform the ready-go task.
Fitness threshold set in config
- by default very high to show the high possible accuracy of this library.
"""

import copy
import sys
import os
import pickle
import math
import argparse
import neat
import neat.nn
import neat.ctrnn
import multiprocessing
import numpy as np
import shutil
from pureples.shared.visualize import draw_net, draw_output, draw_hebbian
from pureples.shared.ready_go import ready_go_list
from pureples.shared.population_plus import Population
from pureples.shared.distributions import bimodal

foreperiod = 25
cycles = 100
time_block_size = 5
cycle_delay_range = [0, 3]
cycle_len = math.floor(foreperiod / time_block_size)
# identity_func = lambda x: x
training_setup = {
    "function": ready_go_list,
    "params": [
        foreperiod,
        cycles,
        time_block_size,
        cycle_delay_range,
        [
            # np.random.normal,
            # np.random.triangular,
            # np.random.triangular,
            # bimodal,
            np.random.normal,
            np.random.normal,
            np.random.normal,
            np.random.normal,
            np.random.normal,
        ],
        [
            # {"loc": math.floor(cycle_len / 2), "scale": cycle_len / 4},
            # {"left": 0, "mode": cycle_len, "right": cycle_len},
            # {"left": 0, "mode": 0, "right": cycle_len},
            # {
            #     "loc": [math.floor(cycle_len / 4), math.ceil(cycle_len * 3 / 4)],
            #     "scale": [cycle_len / 8, cycle_len / 8],
            # },
            {"loc": 0, "scale": 0},
            {"loc": 1, "scale": 0},
            {"loc": 2, "scale": 0},
            {"loc": 3, "scale": 0},
            {"loc": 4, "scale": 0},
        ],
    ],
}

config_file = (
    "pureples/experiments/ready_go/config_neat_ready_go"
    if len(sys.argv) < 6
    else sys.argv[5]
)

# Config for network
CONFIG = neat.config.Config(
    neat.genome.DefaultGenome,
    neat.reproduction.DefaultReproduction,
    neat.species.DefaultSpeciesSet,
    neat.stagnation.DefaultStagnation,
    config_file,
)


def run_network(
    network, net, ready_go_data, verbose=False, visualize="", cycle_len=0, key=None
):
    print_fitness = verbose
    network_fitness = []

    trial = 0

    for inputs, expected_output in ready_go_data:
        outputs = []
        trial += 1
        last_fitness = 0.0
        fitness = []
        steady = False
        training_over = False
        output = [0, 0]
        # net.reset()
        for index, input in enumerate(inputs):
            ready = int(input == 1)
            go = int(input == 2)
            steady = (steady or ready) and not go

            # Do we really even need the Go signal?
            output = net.advance([ready, go], 0.05, 0.01)

            last_fitness = 1 - abs(output[0] - expected_output[index]) ** 2

            outputs.append(output[0])
            if not training_over and ready and index >= 30:
                training_over = True
            if training_over:
                if last_fitness != -1.0:
                    fitness.append(last_fitness)
            # if expected_output[0] < 0.1:
            #     last_fitness = 0.0

            if verbose and training_over:
                print(
                    " input {!r}, expected output {:.3f}, got {!r}".format(
                        input, expected_output[index], output
                    )
                )
        if fitness:
            network_fitness.append(np.mean(fitness))
        else:
            network_fitness.append([0.0])

        if visualize:
            draw_output(
                cycle_len,
                np.array(inputs),
                np.array(outputs),
                np.array(expected_output),
                f"pureples/experiments/ready_go/results/{folder}/hebbian_neat_ready_go_population{key}_{visualize}_{trial}_outputs.png",
            )
            draw_hebbian(
                net.hebbian_update_log,
                f"pureples/experiments/ready_go/results/{folder}/hebbian_neat_ready_go_population{key}_{visualize}_{trial}_hebbian.png",
            )
        verbose = False
        if print_fitness:
            print(net.node_evals)
    if print_fitness:
        print(network_fitness)
    return np.mean(network_fitness)


def _eval_fitness(genome, config):
    """
    Fitness function.
    Evaluate the fitness of a single genome.
    """
    network = neat.ctrnn.CTRNN.create(genome, config, 5)

    # genome fitness
    return run_network(None, network, config.train_set)


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
    pop = ini_pop(initial_pop, CONFIG, stats_one)
    pe = neat.ParallelEvaluator(8, _eval_fitness)
    species_one = pop.run(pe.evaluate, gens)

    # print("Second run")
    # setattr(CONFIG, "trials", distributions)
    # stats_ten = neat.StatisticsReporter()
    # pop = ini_pop((pop.population, pop.species, 0), CONFIG, stats_ten)
    # winner_ten = pop.run(pe.evaluate, gens)

    # if max_trials == 0:
    #     return winner_ten, (stats_one, stats_ten)

    # print("Third run")
    # setattr(CONFIG, "trials", max_trials * distributions)
    # stats_hundred = neat.StatisticsReporter()
    # pop = ini_pop((pop.population, pop.species, 0), CONFIG, stats_hundred)
    # print(f"hebbian_neat_ready_go done")
    # winner_hundred = pop.run(pe.evaluate, gens)
    return species_one, (stats_one)  # , stats_ten, stats_hundred)


def extract_winning_species(species_set):
    species_winners = {k: None for k in species_set.keys()}
    # print(f"{species_winners=}")
    # print(f"{species_set.items()=}")
    for key, species in species_set.items():
        # print(f"{species.get_fitnesses()=}")
        # print(f"{species.members=}")
        # print(f"{key=}")
        # print(f"{dir(species)=}")
        for genome in species.members.values():
            # print(f"{genome=}")
            # print(f"{genome.fitness=}")
            # print(f"{fitness=}")
            # if species_winners[key]:
            #     print(f"{species_winners[key].fitness=}")

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

    return species_winners


# If run as script.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gens", default=1)
    parser.add_argument("--target_folder", default="results")
    parser.add_argument("--load", default=None)
    parser.add_argument(
        "--config", default="pureples/experiments/ready_go/config_neat_ready_go"
    )
    parser.add_argument("--abs", default=False)
    parser.add_argument("--firing_threshold", default=0.2)
    parser.add_argument("--hebbian_learning_rate", default=0.05)
    args = parser.parse_args()

    setattr(CONFIG, "absolute_weights", bool(args.abs))
    setattr(CONFIG, "firing_threshold", float(args.firing_threshold))
    setattr(CONFIG, "learning_rate", float(args.hebbian_learning_rate))

    folder = args.target_folder
    save_dir = os.path.join(os.path.dirname(__file__), f"results/{folder}")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    shutil.copyfile(
        args.config,
        f"pureples/experiments/ready_go/results/{folder}/config_neat_ready_go",
    )
    with open(
        f"pureples/experiments/ready_go/results/{folder}/args.txt",
        "w",
    ) as output:
        output.write(str(args))

    if args.load:
        with open(args.load, "rb") as f:
            WINNERS = {"winner": pickle.load(f)}
    else:
        result = run(gens=int(args.gens))
        winner = result[0][0]  # Only relevant to look at the winner.
        print("\nBest genome:\n{!s}".format(winner))

        WINNERS = extract_winning_species(result[0][2])
    count = 0

    max_trial_len = cycle_len + cycle_delay_range[1]
    test_set = (
        result[0][1]
        if not args.load
        else training_setup["function"](*training_setup["params"])
    )
    test_set_expanded = copy.deepcopy(test_set)
    # for i in [0, cycle_len // 2, cycle_len - 1, -1]:
    #     for j in range(len(test_set)):
    #         test_set_expanded[j][0].append(1)
    #         test_set_expanded[j][1].append(0)
    #         for k in range(max_trial_len - 1):
    #             if i == k:
    #                 test_set_expanded[j][0].append(2)
    #                 test_set_expanded[j][1].append(1)
    #             else:
    #                 test_set_expanded[j][0].append(0)
    #                 test_set_expanded[j][1].append(0)

    for WINNER in WINNERS.values():
        count += 1
        # Verify network output against training data.
        print("\nOutput:")
        NETWORK = neat.ctrnn.CTRNN.create(WINNER, CONFIG, 5)

        NETWORK.reset()
        WINNER.fitness = run_network(
            None,
            NETWORK,
            test_set_expanded,
            verbose=False,
            visualize=f"population{count}_all",
            cycle_len=max_trial_len,  # Assume cycle_len is same/larger for last dist
            key=count,
        )

        # Save network if wished reused and draw it to file.
        draw_net(
            NETWORK,
            filename=f"pureples/experiments/ready_go/results/{folder}/hebbian_neat_ready_go_population{count}_network",
            node_names={
                -1: "ready",
                -2: "go",
                -3: "fitness",
                0: "o",
                1: "m",
            },
            node_colors={
                -1: "yellow",
                -2: "green",
                -3: "grey",
                0: "lightblue",
                1: "lightblue",
            },
        )
        with open(
            f"pureples/experiments/ready_go/results/{folder}/hebbian_neat_ready_go_population{count}_genome.pkl",
            "wb",
        ) as output:
            pickle.dump(WINNER, output, pickle.HIGHEST_PROTOCOL)

        with open(
            f"pureples/experiments/ready_go/results/{folder}/hebbian_neat_ready_go_population{count}_network.pkl",
            "wb",
        ) as output:
            pickle.dump(NETWORK, output, pickle.HIGHEST_PROTOCOL)

        with open(
            f"pureples/experiments/ready_go/results/{folder}/hebbian_neat_ready_go_population{count}_genome.txt",
            "w",
        ) as output:
            output.write(str(WINNER))

# TODO
# # Visualize
## Comparison of output between distributions
## Color-code your outputs?
## Response factor of nodes/connection (which one is it actually?)
### There was something else to add when visualizing nodes, right?

# Hebbian
## Ensure weights are updated correctly according to algorithm
## Would supplying the hebbian updates with the fitness directly be more biologically plausible than the current input-output implementation?
## Is multiplying the hebbian factor with an evolved scalar biologically plausible?
## Make learning rate trainable
## The Backpropamine paper's implementation differs slightly, try theirs as well
# Experiments
## Focus on single-signed, no fitness input, WITH modulation
## Try weights up to 10 (to observe hebbian changes in a more fine-grained environment)
## Implement the experiment from the Maes et al. 2020 paper
