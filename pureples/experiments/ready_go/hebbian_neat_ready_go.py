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
import multiprocessing
import numpy as np
import shutil
import pureples
from pureples.shared.visualize import draw_net, draw_net2, draw_output, draw_hebbian
from pureples.shared.ready_go import ready_go_list
from pureples.shared.population_plus import Population
from pureples.shared.hebbian_rnn import HebbianRecurrentNetwork
from pureples.shared.distributions import bimodal
from pureples.shared.IZNodeGene_plus import IZNN, IZGenome


parser = argparse.ArgumentParser()
parser.add_argument("--gens", default=1)
parser.add_argument("--target_folder", default=None)
parser.add_argument("--load", default=None)
parser.add_argument(
    "--config", default="pureples/experiments/ready_go/config_neat_ready_go"
)
parser.add_argument(
    "--hebbian_type", default="positive", choices=["positive", "signed", "unsigned"]
)  # not yet implemented
parser.add_argument("--binary_weights", default=False)
parser.add_argument("--firing_threshold", default=0.20)
parser.add_argument("--hebbian_learning_rate", default=0.05)
parser.add_argument("--experiment", default="12345-unordered")  # not yet implemented
parser.add_argument("--reset", default=False)
parser.add_argument("--suffix", default="")
parser.add_argument("--model", default="rnn", choices=["rnn", "iznn"])
parser.add_argument("--overwrite", default=False)
parser.add_argument("--end_test", default=False)
args = parser.parse_args()

foreperiod = 25
trials = 100
time_block_size = 5
cycle_delay_range = [0, 3]
cycle_len = math.floor(foreperiod / time_block_size)
# identity_func = lambda x: x
training_setup = {
    "function": ready_go_list,
    "params": [
        foreperiod,
        trials,
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

# Config for network
CONFIG = None
if args.model == "rnn":
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
    network,
    net,
    ready_go_data,
    verbose=False,
    visualize="",
    cycle_len=0,
    key=None,
    end_tests=0,
):
    print_fitness = verbose
    network_fitness = []

    train_set = 0

    for inputs, expected_output in ready_go_data:
        trial = 0
        # trials = len(inputs)
        outputs = []
        all_outputs = []
        train_set += 1
        last_fitness = 0.0
        fitness = []
        steady = False
        training_over = False
        output = [0, 0]
        if args.reset:
            net.reset()
        for index, input in enumerate(inputs):
            ready = int(input == 1)
            go = int(input == 2)
            steady = (steady or ready) and not go

            if ready:
                trial += 1
            end_test = trials - trial > end_tests

            # Do we really even need the Go signal?
            output = net.activate([ready, go], end_test)

            last_fitness = 1 - abs(output[0] - expected_output[index]) ** 2

            outputs.append(output[0])
            all_outputs.append(copy.deepcopy(net.ovalues))
            if not training_over and ready and index >= 30:
                training_over = True
            if training_over and not end_test:
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
                f"pureples/experiments/ready_go/results/{folder_name}/hebbian_neat_ready_go_population{key}/run_{train_set}_outputs.png",
                end_tests=end_tests,
                all_outputs=all_outputs,
            )
            draw_hebbian(
                net.hebbian_update_log,
                f"pureples/experiments/ready_go/results/{folder_name}/hebbian_neat_ready_go_population{key}/run_{train_set}_hebbian.png",
            )
        verbose = False
        if print_fitness:
            print(net.node_evals)
    if print_fitness:
        print(network_fitness)
    return np.mean(network_fitness)


def compute_output(t0, t1):
    """Compute the network's output based on the "time to first spike" of the two output neurons."""
    if t0 is None or t1 is None:
        # If neither of the output neurons fired within the allotted time,
        # give a response which produces a large error.
        return -1.0
    else:
        # If the output neurons fire within 1.0 milliseconds of each other,
        # the output is 1, and if they fire more than 11 milliseconds apart,
        # the output is 0, with linear interpolation between 1 and 11 milliseconds.
        response = 1.1 - 0.1 * abs(t0 - t1)
        return max(0.0, min(1.0, response))


def compute_output_single(spike_timings):
    """Compute the network's output based on the spiking rate of the output neuron."""
    # If the output neuron spikes 50% or more of the time, the output is 1
    # Linear interpolation between 0 and 1
    return min(
        (len(spike_timings) - spike_timings.count(0)) / len(spike_timings) * 2, 1
    )


def run_iznn(
    network,
    net,
    ready_go_data,
    verbose=False,
    visualize="",
    cycle_len=0,
    key=None,
    end_test=False,
):
    print_fitness = verbose
    network_fitness = []
    dt = net.get_time_step_msec()
    max_time_msec = 11

    train_set = 0
    for inputs, expected_output in ready_go_data:
        outputs = []
        train_set += 1
        trial = 0
        last_fitness = 0.0
        fitness = []
        training_over = False
        output = [0, 0]

        neuron_data = {}
        for i, n in net.neurons.items():
            neuron_data[i] = []

        num_steps = int(max_time_msec / dt)
        # net.reset()
        for index, input in enumerate(inputs):
            trial += 1
            ready = int(input == 1)
            go = int(input == 2)
            t0 = None
            t1 = None
            v0 = None
            v1 = None
            net.set_inputs([ready, go])
            spike_timings = []
            for j in range(num_steps):
                t = dt * j
                # Do we really even need the Go signal?
                output = net.advance(dt)

                # Capture the time and neuron membrane potential for later use if desired.
                for i, n in net.neurons.items():
                    neuron_data[i].append((t, n.current, n.v, n.u, n.fired))

                # Remember time and value of the first output spikes from each neuron.
                if t0 is None and output[0] > 0:
                    t0, I0, v0, u0, f0 = neuron_data[net.outputs[0]][-2]

                if t1 is None and output[1] > 0:
                    t1, I1, v1, u1, f0 = neuron_data[net.outputs[1]][-2]
                spike_timings.append(output[0])

            respone = compute_output(t0, t1)
            # respone = compute_output_single(spike_timings)

            last_fitness = 1 - abs(respone - expected_output[index]) ** 2

            outputs.append(respone)
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
                f"pureples/experiments/ready_go/results/{folder_name}/hebbian_neat_ready_go_population{key}/run_{train_set}_outputs.png",
                end_test=end_test,
            )
            draw_hebbian(
                net.hebbian_update_log,
                f"pureples/experiments/ready_go/results/{folder_name}/hebbian_neat_ready_go_population{key}/run_{train_set}_hebbian.png",
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

    if args.model == "rnn":
        network = HebbianRecurrentNetwork.create(genome, config)

        # genome fitness
        return run_rnn(None, network, config.train_set)
    elif args.model == "iznn":
        network = IZNN.create(genome, config)

        # genome fitness
        return run_iznn(None, network, config.train_set)


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
    pe = neat.ParallelEvaluator(16, _eval_fitness)
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
    all_time_best = pop.reporters.reporters[0].best_genome()
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

    hebbian_magnitude = (
        "dynamic"
        if args.model != "iznn" and CONFIG.genome_config.response_mutate_rate
        else "static"
    ) + "_magnitude-hebbian"
    node_weights = "binary_weight" if args.binary_weights else "real_weight"
    reset = "reset" if args.reset else "no-reset"
    folder_name = (
        args.target_folder
        if args.target_folder
        else f"{args.experiment}-{args.model}-{args.hebbian_type}-{hebbian_magnitude}-{('%.2f' % args.firing_threshold).split('.')[1]}-threshold-{node_weights}-{reset}{args.suffix}"
    )

    save_dir = os.path.join(os.path.dirname(__file__), f"results/{folder_name}")
    similar_run = 0
    while os.path.exists(save_dir) and not args.overwrite:
        similar_run += 1
        save_dir = save_dir.split("__")[0] + "__" + str(similar_run)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    folder_name = save_dir.split("results/")[1]

    shutil.copyfile(
        args.config,
        f"pureples/experiments/ready_go/results/{folder_name}/{args.config.split('/')[-1]}",
    )
    with open(
        f"pureples/experiments/ready_go/results/{folder_name}/args.txt",
        "w",
    ) as output:
        output.write(str(args))

    if args.load:
        with open(args.load, "rb") as f:
            WINNERS = {"winner": pickle.load(f)}
    else:
        result = run(gens=int(args.gens))
        winner = result[2]  # All-time best
        print("\nBest genome:\n{!s}".format(winner))

        WINNERS = extract_winning_species(result[0][2], winner)

    count = 0

    max_trial_len = cycle_len + cycle_delay_range[1]
    test_set = (
        result[0][1]
        if not args.load
        else training_setup["function"](*training_setup["params"])
    )
    test_set_expanded = copy.deepcopy(test_set)
    end_tests = int(args.end_test)
    if end_tests:
        for i in [0, cycle_len // 2, cycle_len - 1, -1]:
            for j in range(len(test_set)):
                test_set_expanded[j][0].append(1)
                test_set_expanded[j][1].append(0)
                for k in range(max_trial_len - 1):
                    if i == k:
                        test_set_expanded[j][0].append(2)
                        test_set_expanded[j][1].append(1)
                    else:
                        test_set_expanded[j][0].append(0)
                        test_set_expanded[j][1].append(0)

    for WINNER in WINNERS.values():
        count += 1
        population_dir = f"pureples/experiments/ready_go/results/{folder_name}/hebbian_neat_ready_go_population{count}/"
        if not os.path.exists(population_dir):
            os.mkdir(population_dir)
        # Verify network output against training data.
        print("\nOutput:")
        if args.model == "rnn":
            NETWORK = HebbianRecurrentNetwork.create(WINNER, CONFIG)
            NETWORK.reset()
            WINNER.fitness = run_rnn(
                None,
                NETWORK,
                test_set_expanded,
                verbose=False,
                visualize=f"population{count}_all",
                cycle_len=max_trial_len,  # Assume cycle_len is same/larger for last dist
                key=count,
                end_tests=end_tests,
            )
        elif args.model == "iznn":
            NETWORK = IZNN.create(WINNER, CONFIG)
            NETWORK.reset()
            WINNER.fitness = run_iznn(
                None,
                NETWORK,
                test_set_expanded,
                verbose=False,
                visualize=f"population{count}_all",
                cycle_len=max_trial_len,
                key=count,
                end_test=end_tests,
            )
        # Save network if wished reused and draw it to file.
        with open(
            f"{population_dir}genome.pkl",
            "wb",
        ) as output:
            pickle.dump(WINNER, output, pickle.HIGHEST_PROTOCOL)

        with open(
            f"{population_dir}network.pkl",
            "wb",
        ) as output:
            pickle.dump(NETWORK, output, pickle.HIGHEST_PROTOCOL)

        with open(
            f"{population_dir}genome.txt",
            "w",
        ) as output:
            output.write(str(WINNER))

        draw_net(
            CONFIG,
            WINNER,
            filename=f"{population_dir}network",
            prune_unused=True,
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

# TODO
# Visualize
##^ Output at start and end of each run
##^ Run omission tests (WITHOUT PLASTICITY)
##^ Both above -> ALL NODE OUTPUTS
##* NETWORK
###* Response factor and bias of nodes, weight and hebbian of connections at different timesteps
####^ Response factor and bias of nodes, weight
#### Dotted line for connections with varying hebbian
#### At different timesteps
## Fitness of population over time
### Species over time?

# Hebbian
## Make learning rate trainable
## The Backpropamine paper's implementation differs slightly, try theirs as well

# Experiments
## Focus on POSITIVE, WITH modulation
## Try weights up to 10 (to observe hebbian changes in a more fine-grained environment)
## Implement the experiment from the Maes et al. 2020 paper

# Thesis
## Have to explain HOW/WHY the networks form the topologies
### HOW is the information ENCODED?
### HOW does the networks LEARN?
## Evolution of the networks

# Next meeting
## Make list of potential figures
##
