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
from pureples.shared.visualize import draw_net, draw_hist
from pureples.shared.ready_go import ready_go_list
from pureples.shared.population_plus import Population

# S, M or L; Small, Medium or Large (logic implemented as "Not 'S' or 'M' then Large").
VERSION = "M"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"

foreperiod = 25
cycles = 500
time_block_size = 5
cycle_delay_range = [0, 3]
training_setup = {
    "function": ready_go_list,
    "params": [foreperiod, cycles, time_block_size, cycle_delay_range],
}

# Config for network
CONFIG = neat.config.Config(
    neat.genome.DefaultGenome,
    neat.reproduction.DefaultReproduction,
    neat.species.DefaultSpeciesSet,
    neat.stagnation.DefaultStagnation,
    "pureples/experiments/ready_go/config_neat_ready_go",
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
            draw_hist(
                "",
                np.array(inputs),
                np.array(outputs),
                np.array(expected_output),
                f"pureples/experiments/ready_go/results/neat_ready_go_{VERSION_TEXT}_outputs.png",
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
    network = neat.nn.RecurrentNetwork.create(genome, config)

    # genome fitness
    return run_network(None, network, config.train_set)


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
    print(f"neat_ready_go_{VERSION_TEXT} done")
    winner_hundred = pop.run(pe.evaluate, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)


# If run as script.
if __name__ == "__main__":
    result = run(gens=50, version=VERSION)
    WINNER = result[0][0]  # Only relevant to look at the winner.
    print("\nBest genome:\n{!s}".format(WINNER))

    # Verify network output against training data.
    print("\nOutput:")
    NETWORK = neat.nn.RecurrentNetwork.create(WINNER, CONFIG)
    # Add logic to draw winner_net.
    draw_net(
        NETWORK,
        f"pureples/experiments/ready_go/results/neat_ready_go_{VERSION_TEXT}_winner.png",
    )

    run_network(None, NETWORK, result[0][1], verbose=False, visualize=True)

    # Save CPPN if wished reused and draw it to file.
    draw_net(
        NETWORK,
        filename=f"pureples/experiments/ready_go/results/neat_ready_go_{VERSION_TEXT}_network",
    )
    with open(
        f"pureples/experiments/ready_go/results/neat_ready_go_{VERSION_TEXT}_network.pkl",
        "wb",
    ) as output:
        pickle.dump(NETWORK, output, pickle.HIGHEST_PROTOCOL)
