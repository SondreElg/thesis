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
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat_ctrnn.es_hyperneat_ctrnn import ESNetworkCTRNN
from pureples.shared.ready_go import ready_go_list
from pureples.shared.concurrent_neat_population import Population

# S, M or L; Small, Medium or Large (logic implemented as "Not 'S' or 'M' then Large").
VERSION = "M"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"

# Network coordinates and the resulting substrate.
INPUT_COORDINATES = [(0.0, -1.0)]
OUTPUT_COORDINATES = [(0.0, 1.0)]
SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)


def params(version):
    """
    ES-HyperNEAT specific parameters.
    """
    return {
        "initial_depth": 0 if version == "S" else 1 if version == "M" else 2,
        "max_depth": 1 if version == "S" else 2 if version == "M" else 3,
        "variance_threshold": 0.8,
        "band_threshold": 0.9,
        "iteration_level": 1,
        "division_threshold": 0.5,
        "max_weight": 5.0,
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


def run_network(network, net, verbose=False):
    net.reset()
    fitness = 0.0
    expected = 0

    foreperiod = 30
    cycles = 100
    time_block_size = 5
    ready_go_inputs = ready_go_list(foreperiod, cycles, time_block_size, cycle_delay=5)
    predicted = False
    ready = 0
    go = 0
    state = 0
    distance = 999
    for index, ready_go_input in enumerate(ready_go_inputs):
        go = int(ready_go_input == 2)
        state = 2 if go else 1 if (ready or state == 1) else 0
        ready = int(ready_go_input == 1)
        if ready_go_input == 1:
            expected = 0
        # net.reset()

        # if ready:
        #     predicted = False
        # elif state == 2:
        #     predicted = True

        # Why does it activate multiple times?
        # Does the recurrent network implementation progress one "layer" at a time?
        # for _ in range(network.activations):
        # Do we really even need the Go signal?
        output = net.advance([ready], 5, 1)

        # expected = 0
        # if state == 0:
        #     fitness -= (abs(output[0] - expected)) ** 2
        # else:
        #     if state == 1:
        #         distance = ready_go_inputs.index(2, index) - index + 1
        #         expected = 1 / distance**2
        #     elif state == 2:
        #         distance += 1
        #         expected = 1 / distance**2
        #     fitness += (1 - abs(output[0] - expected)) ** 2
        if state == 1:
            distance = ready_go_inputs.index(2, index) - index + 2
            expected += 1 / distance**2
        else:
            if state == 2:
                distance = 0
            distance += 1
            expected = 1 / distance**2
        fitness += (1 - abs(output[0] - expected)) ** 2

        # if len(ready_go_inputs) - index < 10:
        if verbose:
            print(
                "  input {!r}, expected output {:.3f}, got {!r}".format(
                    ready_go_input, expected, output
                )
            )
    # print("\n")

    return fitness / len(ready_go_inputs)


def _eval_fitness(genome, config):
    """
    Fitness function.
    Evaluate the fitness of a single genome.
    """
    cppn = neat.nn.FeedForwardNetwork.create(genome, config)
    network = ESNetworkCTRNN(SUBSTRATE, cppn, DYNAMIC_PARAMS)
    net = network.create_phenotype_network()

    # genome fitness
    return run_network(network, net)


def eval_fitness(genomes, config):
    """
    Fitness function.
    For each genome evaluate its fitness.
    """
    for _, genome in genomes:
        genome.fitness = _eval_fitness(genome, config)


def ini_pop(state, config, stats):
    pop = neat.Population(config, state)
    add_pop_reporter(pop, stats)
    return pop


def add_pop_reporter(pop, stats):
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True))


def run(*, gens, version, max_trials=100):
    """
    Create the population and run the ready_go task by providing eval_fitness as the fitness function.
    Returns the winning genome and the statistics of the run.
    """

    global DYNAMIC_PARAMS
    DYNAMIC_PARAMS = params(version)

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.StatisticsReporter()
    pop = ini_pop(None, CONFIG, stats_one)
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), _eval_fitness)
    pop.run(pe.evaluate, gens)

    stats_ten = neat.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), CONFIG, stats_ten)
    trials = 10
    winner_ten = pop.run(pe.evaluate, gens)

    if max_trials == 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), CONFIG, stats_hundred)
    trials = max_trials
    print(f"es_hyperneat_ctrnn_ready_go_{VERSION_TEXT} done")
    winner_hundred = pop.run(pe.evaluate, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)


# If run as script.
if __name__ == "__main__":
    WINNER = run(gens=100, version=VERSION)[0]  # Only relevant to look at the winner.
    print("\nBest genome:\n{!s}".format(WINNER))

    # Verify network output against training data.
    print("\nOutput:")
    CPPN = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)
    NETWORK = ESNetworkCTRNN(SUBSTRATE, CPPN, DYNAMIC_PARAMS)
    # This will also draw winner_net.
    WINNER_NET = NETWORK.create_phenotype_network(
        filename=f"pureples/experiments/ready_go/results/es_hyperneat_ctrnn_ready_go_{VERSION_TEXT}_winner.png"
    )

    run_network(NETWORK, WINNER_NET, verbose=True)

    # Save CPPN if wished reused and draw it to file.
    draw_net(
        CPPN,
        filename=f"pureples/experiments/ready_go/results/es_hyperneat_ctrnn_ready_go_{VERSION_TEXT}_cppn",
    )
    with open(
        f"pureples/experiments/ready_go/results/es_hyperneat_ctrnn_ready_go_{VERSION_TEXT}_cppn.pkl",
        "wb",
    ) as output:
        pickle.dump(CPPN, output, pickle.HIGHEST_PROTOCOL)
