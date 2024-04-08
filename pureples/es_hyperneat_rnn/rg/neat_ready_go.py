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
from pureples.shared.visualize import draw_net, draw_output
from pureples.shared.ready_go import ready_go_list, ready_go_list_zip
from pureples.shared.no_direct_rnn import RecurrentNetwork
from pureples.shared.population_plus import Population

# S, M or L; Small, Medium or Large (logic implemented as "Not 'S' or 'M' then Large").
VERSION = "M"
VERSION_TEXT = (
    "small" if VERSION == "S" else "rising_triangle" if VERSION == "M" else "long"
)

foreperiod = 25
cycles = 500
time_block_size = 5
cycle_delay_range = [0, 3]
cycle_len = math.floor(foreperiod / time_block_size)
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
            np.random.triangular,
            #
        ],
        [
            # {"loc": math.floor(cycle_len / 2), "scale": cycle_len / 4},
            {"left": 0, "mode": cycle_len - 1, "right": cycle_len - 1},
            # {"left": 0, "mode": 0, "right": cycle_len - 1},
        ],
    ],
}

# Config for network
CONFIG = neat.config.Config(
    neat.genome.DefaultGenome,
    neat.reproduction.DefaultReproduction,
    neat.species.DefaultSpeciesSet,
    neat.stagnation.DefaultStagnation,
    "pureples/experiments/ready_go/config_neat_ready_go",
)


def run_network(network, net, ready_go_data, verbose=False, visualize="", cycle_len=0):
    total_entries = 0
    outputs = []
    print_fitness = verbose
    trial_fitness = []

    for inputs, expected_output in ready_go_data:
        last_fitness = 0.0
        fitness = []
        steady = False
        training_over = False
        net.reset()
        for index, input in enumerate(inputs):
            ready = int(input == 1)
            go = int(input == 2)
            steady = (steady or ready) and not go

            output = net.activate([ready, go])

            last_fitness = 1 - abs(output[0] - expected_output[index]) ** 2

            # if not training_over and ready and index >= len(inputs) // 2:
            #     training_over = True
            # if training_over:
            outputs.append(*output)
            fitness.append(last_fitness)
            # if last_fitness != -1.0:

            if verbose:
                print(
                    " input {!r}, expected output {:.3f}, got {!r}".format(
                        input, expected_output[index], output
                    )
                )
        if fitness:
            trial_fitness.append(np.mean(fitness))
        else:
            trial_fitness.append(0.0)

        if visualize:
            draw_output(
                cycle_len,
                np.array(inputs),
                np.array(outputs),
                np.array(expected_output),
                f"pureples/experiments/ready_go/results/neat_ready_go_{VERSION_TEXT}_{visualize}_outputs.png",
            )
            visualize = ""
        verbose = False
        total_entries += len(inputs)
    #     if print_fitness:
    #         print(net.node_evals)
    # if print_fitness:
    #     print(trial_fitness)
    return np.mean(trial_fitness)


def _eval_fitness(genome, config):
    """
    Fitness function.
    Evaluate the fitness of a single genome.
    """
    network = RecurrentNetwork.create(genome, config)

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


def run(*, gens, version, max_trials=1, initial_pop=None):
    """
    Create the population and run the ready_go task by providing eval_fitness as the fitness function.
    Returns the winning genome and the statistics of the run.
    """

    setattr(CONFIG, "training_setup", training_setup)

    distributions = len(training_setup["params"][4])

    # Create population and train the network. Return winner of network running 100 episodes.
    print("First run")
    setattr(CONFIG, "trials", distributions)
    stats_one = neat.StatisticsReporter()
    pop = ini_pop(initial_pop, CONFIG, stats_one)
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count() - 1, _eval_fitness)
    winner_one = pop.run(pe.evaluate, gens)

    return winner_one, (stats_one)


# If run as script.
if __name__ == "__main__":
    result = run(gens=200, version=VERSION)
    WINNER = result[0][0]  # Only relevant to look at the winner.
    print("\nBest genome:\n{!s}".format(WINNER))

    # Verify network output against training data.
    print("\nOutput:")
    NETWORK = RecurrentNetwork.create(WINNER, CONFIG)
    # Add logic to draw winner_net.
    distributions = len(training_setup["params"][4])

    max_cycle_len = (
        math.floor(training_setup["params"][0] / training_setup["params"][2])
        + training_setup["params"][3][1]
    )

    if distributions > 1:
        for i in range(distributions):
            single_dist_setup = {
                "function": training_setup["function"],
                "params": [
                    *training_setup["params"][:4],
                    [training_setup["params"][4][i]],
                    [training_setup["params"][5][i]],
                ],
            }
            cycle_len = (
                math.floor(
                    single_dist_setup["params"][0] / single_dist_setup["params"][2]
                )
                + single_dist_setup["params"][3][1]
            )
            max_cycle_len = max(max_cycle_len, cycle_len)
            setattr(CONFIG, "training_setup", single_dist_setup)
            NETWORK = RecurrentNetwork.create(WINNER, CONFIG)
            data = single_dist_setup["function"](*single_dist_setup["params"])
            run_network(
                None,
                NETWORK,
                data,
                verbose=True,
                visualize=f"{VERSION_TEXT}_dist{i+1}",
                cycle_len=cycle_len,
            )

    run_network(
        None,
        NETWORK,
        result[0][1],
        verbose=False,
        visualize=f"{VERSION_TEXT}_all",
        cycle_len=max_cycle_len,  # Assume cycle_len is same/larger for last dist
    )

    # run_network(None, NETWORK, result[0][1], verbose=False, visualize=True)

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
