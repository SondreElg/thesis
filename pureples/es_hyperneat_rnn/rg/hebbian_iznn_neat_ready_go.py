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
from pureples.shared.visualize import draw_net, draw_output, draw_hebbian
from pureples.shared.ready_go import ready_go_list
from pureples.shared.population_plus import Population
from pureples.shared.distributions import bimodal
from pureples.shared.IZNodeGene_plus import IZGenome

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
args = parser.parse_args()

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

# Config for network
CONFIG = neat.config.Config(
    neat.iznn.IZGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    args.config,
)


def run_network(
    network, net, ready_go_data, verbose=False, visualize="", cycle_len=0, key=None
):
    print_fitness = verbose
    network_fitness = []
    dt = net.get_time_step_msec()

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
            output = net.activate([ready, go], 1)

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
                f"pureples/experiments/ready_go/results/{folder_name}/hebbian_neat_ready_go_population{key}_{visualize}_{trial}_outputs.png",
            )
            draw_hebbian(
                net.hebbian_update_log,
                f"pureples/experiments/ready_go/results/{folder_name}/hebbian_neat_ready_go_population{key}_{visualize}_{trial}_hebbian.png",
            )
        verbose = False
        if print_fitness:
            print(net.node_evals)
    if print_fitness:
        print(network_fitness)
    return np.mean(network_fitness)


# Moved here from hebbian_iznn_neat_ready_go.py to save space, as it will stay unused in the foreseeable future
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
    genome,
    net,
    ready_go_data,
    visualize="",
    cycle_len=0,
    key=None,
    end_test=False,
    log_level=0,
):
    network_fitness = []
    dt = net.get_time_step_msec()
    max_time_msec = 11

    block = 0
    for inputs, expected_output in ready_go_data:
        outputs = []
        block += 1
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
                f"{save_dir}/hebbian_neat_ready_go_network{key}/run_{block}_outputs.png",
                end_test=end_test,
            )
            draw_hebbian(
                net.hebbian_update_log,
                f"{save_dir}/hebbian_neat_ready_go_network{key}/run_{block}_hebbian.png",
            )
    return np.mean(network_fitness)


# End of moved code


def _eval_fitness(genome, config):
    """
    Fitness function.
    Evaluate the fitness of a single genome.
    """
    network = neat.iznn.IZNN.create(genome, config)

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
    setattr(CONFIG, "absolute_weights", bool(args.abs))
    setattr(CONFIG, "firing_threshold", float(args.firing_threshold))
    setattr(CONFIG, "learning_rate", float(args.hebbian_learning_rate))

    hebbian_magnitude = (
        "dynamic" if CONFIG.genome_config.response_mutate_rate else "static"
    ) + "_magnitude-hebbian"
    node_weights = "binary_weight" if args.binary_weights else "real_weight"
    reset = "reset" if args.reset else "no-reset"
    folder_name = (
        args.target_folder
        if args.target_folder
        else f"{args.experiment}-{args.hebbian_type}-{hebbian_magnitude}-{('%.2f' % args.firing_threshold).split('.')[1]}-threshold-{node_weights}-{reset}{args.suffix}"
    )

    save_dir = os.path.join(os.path.dirname(__file__), f"results/{folder_name}")
    similar_run = 0
    while os.path.exists(save_dir):
        similar_run += 1
        save_dir = save_dir.split("__")[0] + "__" + str(similar_run)
    os.mkdir(save_dir)

    shutil.copyfile(
        args.config,
        f"pureples/experiments/ready_go/results/{folder_name}/config_neat_ready_go",
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
        NETWORK = neat.iznn.IZNN.create(WINNER, CONFIG)

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
            filename=f"pureples/experiments/ready_go/results/{folder_name}/hebbian_neat_ready_go_population{count}_network",
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
            f"pureples/experiments/ready_go/results/{folder_name}/hebbian_neat_ready_go_population{count}_genome.pkl",
            "wb",
        ) as output:
            pickle.dump(WINNER, output, pickle.HIGHEST_PROTOCOL)

        with open(
            f"pureples/experiments/ready_go/results/{folder_name}/hebbian_neat_ready_go_population{count}_network.pkl",
            "wb",
        ) as output:
            pickle.dump(NETWORK, output, pickle.HIGHEST_PROTOCOL)

        with open(
            f"pureples/experiments/ready_go/results/{folder_name}/hebbian_neat_ready_go_population{count}_genome.txt",
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
