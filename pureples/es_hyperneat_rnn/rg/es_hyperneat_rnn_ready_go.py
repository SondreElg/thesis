"""
An experiment using a variable-sized ES-HyperNEAT network to perform the ready-go task.
Fitness threshold set in config
- by default very high to show the high possible accuracy of this library.
"""

import pickle
import math
import neat
import neat.nn
import multiprocessing
import numpy as np
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net, draw_output
from pureples.es_hyperneat_rnn.es_hyperneat_rnn import ESNetworkRNN
from pureples.shared.ready_go import ready_go_list
from pureples.shared.population_plus import Population

# S, M or L; Small, Medium or Large (logic implemented as "Not 'S' or 'M' then Large").
VERSION = "M"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"

# Network coordinates and the resulting substrate.
INPUT_COORDINATES = [(0.0, -1.0)]  # [(-0.5, -1.0), (0.5, -1.0)]
OUTPUT_COORDINATES = [(0.0, 1.0)]
SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)

foreperiod = 25
cycles = 500
time_block_size = 5
cycle_delay_range = [0, 3]
training_setup = {
    "function": ready_go_list,
    "params": [foreperiod, cycles, time_block_size, cycle_delay_range],
}


def params(version):
    """
    ES-HyperNEAT specific parameters.
    """
    return {
        "initial_depth": 0 if version == "S" else 1 if version == "M" else 2,
        "max_depth": 1 if version == "S" else 2 if version == "M" else 3,
        "variance_threshold": 0.2,
        "band_threshold": 0.3,
        "iteration_level": 3,
        "division_threshold": 0.5,
        "max_weight": 10.0,
        "activation": "sigmoid",
    }


DYNAMIC_PARAMS = params(VERSION)

# Config for CPPN.
CONFIG = neat.config.Config(
    neat.genome.DefaultGenome,
    neat.reproduction.DefaultReproduction,
    neat.species.DefaultSpeciesSet,
    neat.stagnation.DefaultStagnation,
    "pureples/experiments/ready_go/config_cppn_ready_go",
)


def run_network(network, net, ready_go_data, verbose=False, visualize=False):
    fitness = 0.0
    total_entries = 0
    preparatory_entries = 0
    outputs = []
    for inputs, expected_output in ready_go_data:
        cycle_count = 0
        net.reset()
        for index, input in enumerate(inputs):
            ready = int(input == 1)
            go = int(input == 2)
            cycle_count += int(input == 1)

            # if ready:
            #     predicted = False
            # elif state == 2:
            #     predicted = True

            # Why does it activate multiple times?
            # Does the recurrent network implementation progress one "layer" at a time?
            # for _ in range(network.activations):
            # Do we really even need the Go signal?
            output = net.activate([ready])

            # diff = expected - abs(output[0] - expected)
            # # if diff > 0:
            # fitness += diff**2
            if cycle_count > 2:
                if cycle_count == 3 and ready:
                    preparatory_entries += index + 1
                fitness += 1 - abs(output[0] - expected_output[index]) ** 2

            if verbose:
                print(
                    " input {!r}, expected output {:.3f}, got {!r}".format(
                        input, expected_output[index], output
                    )
                )
            outputs.append(*output)

        if visualize:
            draw_output(
                "",
                np.array(inputs),
                np.array(outputs),
                np.array(expected_output),
                f"pureples/experiments/ready_go/results/es_hyperneat_rnn_ready_go_{VERSION_TEXT}_outputs.png",
            )
            visualize = False
        verbose = False
        total_entries += len(inputs)
    return fitness / (total_entries - preparatory_entries)


def _eval_fitness(genome, config):
    """
    Fitness function.
    Evaluate the fitness of a single genome.
    """
    cppn = neat.nn.FeedForwardNetwork.create(genome, config)
    network = ESNetworkRNN(SUBSTRATE, cppn, DYNAMIC_PARAMS)
    net = network.create_phenotype_network()

    # genome fitness
    return run_network(network, net, config.train_set)


def eval_fitness(genomes, config):
    """
    Fitness function.
    For each genome evaluate its fitness.
    """
    # If I want multiple trials per network per generation, I need to find another solution
    ready_go_data = ready_go_list(**training_setup["params"])
    setattr(config, "train_set", ready_go_data)
    for _, genome in genomes:
        genome.fitness = _eval_fitness(genome, config)


def ini_pop(state, config, stats):
    pop = Population(config, state)
    add_pop_reporter(pop, stats)
    return pop


def add_pop_reporter(pop, stats):
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True))


def run(*, gens, version, max_trials=1):
    """
    Create the population and run the ready_go task by providing eval_fitness as the fitness function.
    Returns the winning genome and the statistics of the run.
    """

    global DYNAMIC_PARAMS
    DYNAMIC_PARAMS = params(version)

    setattr(CONFIG, "training_setup", training_setup)

    # Create population and train the network. Return winner of network running 100 episodes.
    setattr(CONFIG, "trials", 1)
    stats_one = neat.StatisticsReporter()
    pop = ini_pop(None, CONFIG, stats_one)
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), _eval_fitness)
    pop.run(pe.evaluate, gens)

    setattr(CONFIG, "trials", 1)
    stats_ten = neat.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), CONFIG, stats_ten)
    winner_ten = pop.run(pe.evaluate, gens)

    if max_trials == 0:
        return winner_ten, (stats_one, stats_ten)

    setattr(CONFIG, "trials", max_trials)
    stats_hundred = neat.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), CONFIG, stats_hundred)
    print(f"es_hyperneat_rnn_ready_go_{VERSION_TEXT} done")
    winner_hundred = pop.run(pe.evaluate, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)


# If run as script.
if __name__ == "__main__":
    result = run(gens=500, version=VERSION)
    WINNER = result[0][0]  # Only relevant to look at the winner.
    print("\nBest genome:\n{!s}".format(WINNER))

    # Verify network output against training data.
    print("\nOutput:")
    CPPN = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)
    NETWORK = ESNetworkRNN(SUBSTRATE, CPPN, DYNAMIC_PARAMS)
    # This will also draw winner_net.
    WINNER_NET = NETWORK.create_phenotype_network(
        filename=f"pureples/experiments/ready_go/results/es_hyperneat_rnn_ready_go_{VERSION_TEXT}_winner.png"
    )

    run_network(NETWORK, WINNER_NET, result[0][1], verbose=False, visualize=True)

    # Save CPPN if wished reused and draw it to file.
    draw_net(
        CPPN,
        filename=f"pureples/experiments/ready_go/results/es_hyperneat_rnn_ready_go_{VERSION_TEXT}_cppn",
    )
    with open(
        f"pureples/experiments/ready_go/results/es_hyperneat_rnn_ready_go_{VERSION_TEXT}_cppn.pkl",
        "wb",
    ) as output:
        pickle.dump(CPPN, output, pickle.HIGHEST_PROTOCOL)
