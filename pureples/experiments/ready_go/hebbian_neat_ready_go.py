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
from pureples.shared.visualize import draw_net, draw_hist, draw_hebbian
from pureples.shared.ready_go import ready_go_list
from pureples.shared.population_plus import Population
from pureples.shared.hebbian_rnn import HebbianRecurrentNetwork
from pureples.shared.distributions import bimodal

# S, M or L; Small, Medium or Large (logic implemented as "Not 'S' or 'M' then Large").
VERSION = "M"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"

foreperiod = 25
cycles = 200
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
            np.random.normal,
            np.random.triangular,
            np.random.triangular,
            bimodal,
            # np.random.normal,
        ],
        [
            {"loc": math.floor(cycle_len / 2), "scale": cycle_len / 4},
            {"left": 0, "mode": cycle_len - 1, "right": cycle_len},
            {"left": 0, "mode": 0, "right": cycle_len},
            {
                "loc": [math.floor(cycle_len / 4), math.ceil(cycle_len * 3 / 4)],
                "scale": [cycle_len / 8, cycle_len / 8],
            },
            # {"loc": math.floor(cycle_len / 2), "scale": 0},
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
        hebbian_factors = []
        net.reset()
        for index, input in enumerate(inputs):
            ready = int(input == 1)
            go = int(input == 2)
            steady = (steady or ready) and not go

            # Do we really even need the Go signal?
            output = net.activate([ready, go, last_fitness], last_fitness)

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
            draw_hist(
                cycle_len,
                np.array(inputs),
                np.array(outputs),
                np.array(expected_output),
                f"pureples/experiments/ready_go/results/hebbian_neat_ready_go_{VERSION_TEXT}_{visualize}_{trial}_outputs.png",
            )
            draw_hebbian(
                net.hebbian_update_log,
                f"pureples/experiments/ready_go/results/hebbian_neat_ready_go_{VERSION_TEXT}_{visualize}_{trial}_hebbian.png",
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
    network = HebbianRecurrentNetwork.create(genome, config)

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


def run(*, gens, version, max_trials=1, initial_pop=None):
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
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count() - 1, _eval_fitness)
    winner_one = pop.run(pe.evaluate, gens)

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
    # print(f"hebbian_neat_ready_go_{VERSION_TEXT} done")
    # winner_hundred = pop.run(pe.evaluate, gens)
    return winner_one, (stats_one)  # , stats_ten, stats_hundred)


# If run as script.
if __name__ == "__main__":
    result = run(gens=100, version=VERSION)
    WINNER = result[0][0]  # Only relevant to look at the winner.
    print("\nBest genome:\n{!s}".format(WINNER))

    # Verify network output against training data.
    print("\nOutput:")
    NETWORK = HebbianRecurrentNetwork.create(WINNER, CONFIG)

    distributions = len(training_setup["params"][4])

    max_trial_len = cycle_len + cycle_delay_range[1]

    # if distributions > 1:
    #     for i in range(distributions):
    #         single_dist_setup = {
    #             "function": training_setup["function"],
    #             "params": [
    #                 *training_setup["params"][:4],
    #                 [training_setup["params"][4][i]],
    #                 [training_setup["params"][5][i]],
    #             ],
    #         }
    #         cycle_len = (
    #             math.floor(
    #                 single_dist_setup["params"][0] / single_dist_setup["params"][2]
    #             )
    #             + single_dist_setup["params"][3][1]
    #         )
    #         max_trial_len = max(max_trial_len, cycle_len)
    #         setattr(CONFIG, "training_setup", single_dist_setup)
    #         NETWORK = HebbianRecurrentNetwork.create(WINNER, CONFIG)
    #         data = single_dist_setup["function"](*single_dist_setup["params"])
    #         run_network(
    #             None,
    #             NETWORK,
    #             data,
    #             verbose=True,
    #             visualize=f"{VERSION_TEXT}_dist{i+1}",
    #             cycle_len=cycle_len,
    #         )

    test_set_expanded = result[0][1]
    for i in [0, cycle_len // 2, cycle_len - 1, -1]:
        for j in range(len(result[0][1])):
            test_set_expanded[j][0].append(1)
            test_set_expanded[j][1].append(0)
            for k in range(max_trial_len - 1):
                if i == k:
                    test_set_expanded[j][0].append(2)
                    test_set_expanded[j][1].append(1)
                else:
                    test_set_expanded[j][0].append(0)
                    test_set_expanded[j][1].append(0)

    # print(f"{test_set_expanded[0][0][-max_trial_len*4:]=}")

    run_network(
        None,
        NETWORK,
        test_set_expanded,
        verbose=False,
        visualize=f"{VERSION_TEXT}_all",
        cycle_len=max_trial_len,  # Assume cycle_len is same/larger for last dist
    )

    # Save network if wished reused and draw it to file.
    draw_net(
        NETWORK,
        filename=f"pureples/experiments/ready_go/results/hebbian_neat_ready_go_{VERSION_TEXT}_network",
    )
    with open(
        f"pureples/experiments/ready_go/results/hebbian_neat_ready_go_{VERSION_TEXT}_network.pkl",
        "wb",
    ) as output:
        pickle.dump(NETWORK, output, pickle.HIGHEST_PROTOCOL)

# TODO
#! Ensure visualization is correct
#! Implement the experiment from the Maes et al. 2020 paper to sanity-check the Ready-Go experiment when your arm doesn't hurt like hell
# Visualize comparison of output between distributions
# Hebbian
## Ensure weights are updated correctly according to algorithm
## Subtract hebbian firing threshold from hebbian update calculations
## Find a way to balance the Hebbian updates
### Current implementation worked in a paper, but seems lackluster when run here
