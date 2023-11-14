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
import json
import numpy as np
import shutil
import pureples
from pureples.shared.visualize import (
    draw_net,
    draw_output,
    draw_hebbian,
    draw_omission_trials,
)
from pureples.shared.ready_go import ready_go_list, foreperiod_rg_list
from pureples.shared.population_plus import Population
from pureples.shared.hebbian_rnn import HebbianRecurrentNetwork
from pureples.shared.hebbian_rnn_plus import HebbianRecurrentDecayingNetwork
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
parser.add_argument("--experiment", default="foreperiod")  # not yet implemented
parser.add_argument("--max_foreperiod", default=25)
parser.add_argument("--foreperiods", default="[1, 2, 3, 4, 5]")
parser.add_argument("--ordering", default="[]")
parser.add_argument("--reset", default=False)
parser.add_argument("--suffix", default="")
parser.add_argument("--model", default="rnn", choices=["rnn", "iznn", "rnn_d"])
parser.add_argument("--overwrite", default=False)
parser.add_argument("--end_test", default=False)
parser.add_argument("--flip_pad_data", default=True)
parser.add_argument("--trial_delay_range", default="[0, 3]")
args = parser.parse_args()

foreperiod = int(args.max_foreperiod)
trials = 50
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
        # [
        #     # np.random.normal,
        #     # np.random.triangular,
        #     # np.random.triangular,
        #     # bimodal,
        #     np.random.normal,
        #     np.random.normal,
        #     np.random.normal,
        #     np.random.normal,
        #     np.random.normal,
        # ],
        # [
        #     # {"loc": math.floor(cycle_len / 2), "scale": cycle_len / 4},
        #     # {"left": 0, "mode": cycle_len, "right": cycle_len},
        #     # {"left": 0, "mode": 0, "right": cycle_len},
        #     # {
        #     #     "loc": [math.floor(cycle_len / 4), math.ceil(cycle_len * 3 / 4)],
        #     #     "scale": [cycle_len / 8, cycle_len / 8],
        #     # },
        #     {"loc": 1, "scale": 0},
        #     {"loc": 2, "scale": 0},
        #     {"loc": 3, "scale": 0},
        #     {"loc": 4, "scale": 0},
        #     {"loc": 5, "scale": 0},
        # ],
        args.flip_pad_data,
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
):
    block = 0
    blocks = len(ready_go_data)

    network_fitness = np.empty((blocks))
    omission_trial_outputs = np.empty((blocks, cycle_len), dtype=object)
    foreperiods = np.empty((blocks), dtype=int)

    for inputs, expected_output in ready_go_data:
        trial = 0
        timesteps = len(inputs)
        outputs = np.empty(timesteps)
        all_outputs = np.empty(timesteps, dict)
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
            if training == -1:
                network = copy.deepcopy(net)

            output = network.activate([ready, go], training)

            last_fitness = 1 - abs(output[0] - expected_output[index]) ** 2
            outputs[index] = output[0]
            all_outputs[index] = copy.deepcopy(network.ovalues)
            if trial >= 6 and training:
                fitness[index] = last_fitness
        network_fitness[block] = np.nanmean(fitness)
        if visualize:
            foreperiod = np.where(np.array(inputs) == 2)[0][0]
            foreperiods[block] = foreperiod - 1
            if end_tests:
                indices = np.asarray(np.where(np.array(inputs) == 1))[0]
                omission_trial_outputs[block] = outputs[
                    indices[-end_tests] : indices[-end_tests + 1]
                ]
            draw_hebbian(
                net,
                f"pureples/experiments/ready_go/results/{folder_name}/hebbian_neat_ready_go_population{key}/block{block+1}_fp{foreperiod}_hebbian.png",
                node_names={
                    -1: "ready",
                    -2: "go",
                    0: "output",
                },
            )
            draw_output(
                cycle_len,
                np.array(inputs),
                np.array(outputs),
                np.array(expected_output),
                f"pureples/experiments/ready_go/results/{folder_name}/hebbian_neat_ready_go_population{key}/block{block+1}_fp{foreperiod}_outputs.png",
                end_tests=end_tests,
                all_outputs=all_outputs,
                network={
                    "config": CONFIG,
                    "genome": genome,
                    "hebbians": net.hebbian_update_log,
                },
                draw_std=True,
            )
        block += 1
    if visualize and end_tests:
        blocks_of_interest = 1 + blocks // 2 if args.flip_pad_data else blocks
        draw_omission_trials(
            omission_trial_outputs[np.argsort(foreperiods[:blocks_of_interest])],
            f"pureples/experiments/ready_go/results/{folder_name}/hebbian_neat_ready_go_population{key}/omission_trials.png",
        )
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
    genome,
    net,
    ready_go_data,
    visualize="",
    cycle_len=0,
    key=None,
    end_test=False,
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
                f"pureples/experiments/ready_go/results/{folder_name}/hebbian_neat_ready_go_population{key}/run_{block}_outputs.png",
                end_test=end_test,
            )
            draw_hebbian(
                net.hebbian_update_log,
                f"pureples/experiments/ready_go/results/{folder_name}/hebbian_neat_ready_go_population{key}/run_{block}_hebbian.png",
            )
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
    pe = neat.ParallelEvaluator(8, _eval_fitness)
    species_one = pop.run(pe.evaluate, gens)

    all_time_best = pop.reporters.reporters[0].best_genome()
    pop.reporters.reporters[0].save_genome_fitness(
        filename=f"pureples/experiments/ready_go/results/{folder_name}/pop_fitness_history.csv",
        delimiter=",",
    )
    pop.reporters.reporters[0].save_species_count(
        filename=f"pureples/experiments/ready_go/results/{folder_name}/pop_speciation_history.csv",
        delimiter=",",
    )
    pop.reporters.reporters[0].save_species_fitness(
        filename=f"pureples/experiments/ready_go/results/{folder_name}/pop_species_fitness_history.csv",
        delimiter=",",
        null_value=-1,
    )
    # Save population if wished reused and draw it to file.
    with open(
        f"pureples/experiments/ready_go/results/{folder_name}/population.pkl",
        "wb",
    ) as output:
        pickle.dump(pop, output, pickle.HIGHEST_PROTOCOL)

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
    if os.path.exists(save_dir) and args.overwrite:
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

    for WINNER in WINNERS.values():
        count += 1
        population_dir = f"pureples/experiments/ready_go/results/{folder_name}/hebbian_neat_ready_go_population{count}/"
        if not os.path.exists(population_dir):
            os.mkdir(population_dir)
        # Verify network output against training data.
        print("\nOutput:")
        if args.model == "rnn" or args.model == "rnn_d":
            NETWORK = (
                HebbianRecurrentNetwork.create(WINNER, CONFIG)
                if args.model == "rnn"
                else HebbianRecurrentDecayingNetwork.create(WINNER, CONFIG)
            )
            NETWORK.reset()
            WINNER.fitness = run_rnn(
                WINNER,
                NETWORK,
                test_set_expanded,
                visualize=f"population{count}_all",
                cycle_len=max_trial_len,  # Assume cycle_len is same/larger for last dist
                key=count,
                end_tests=end_tests,
            )
        elif args.model == "iznn":
            NETWORK = IZNN.create(WINNER, CONFIG)
            NETWORK.reset()
            WINNER.fitness = run_iznn(
                WINNER,
                NETWORK,
                test_set_expanded,
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
## Fitness of population over time
### Size of individuals over time?
## Correlation of hebbian values and foreperiod
#### NMF
#### GLR
#### GLM

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
## Activity of each neuron for each foreperiod

# Thesis
## Have to explain HOW/WHY the networks form the topologies
### HOW is the information ENCODED?
### HOW does the networks LEARN?
## Evolution of the networks
### Background theory
#### Bayesian logic?
## Research goals
### "How can NEAT be used to evolve biologically plausible and interpretable neural networks capable of temporal prediction?"
#### Very long. Too specific?
### "How does RNNs encode prediction over time for temporal prediction tasks?"
#### Very open-ended. Encode hazard function? -> Assumes hazard function will appear from the start
### "How does Hebbian weights correlate to different foreperiods?"
#### "How does Hebbian learning work" is already fairly well understood. Approach from another angle?
## Compare model to another existing model
### Ensure to communicate how your model is unique
## Human model of Ready-Go
## Could the networks resemble those theorized in the paper on predictive coding? [...] canonical circuits

# Reformatting
## Either explain all three networks, or only one
### Could put B and C in appendix
## Fix margin size

# 2
## Cite Burkitt Hodendoorn paper (2019) somewhere?

# 3
## Their network is top-down, mine is bottom-up
## What are you trying to solve?
## See if there's some related work in their introduction
### Introduce that first, and then how Maes builds on it
# 4
## Alpha used for hebbian scaling factor. Reuse symbol
## Rewrite part about "action/axon potential"
## No holdover of node activity between timesteps
# 5
## Use Network A for whole 5.1 section?
### Fig 5.1 only A
### Fig 5.5 all networks + std networks
### Fig 5.6 all networks
## 5.3
### Don't need to analyze networks, just give a theory - explain your thoughts clearly
### What *exactly* do you need to do to interpret the results?
### What do you expect the results to be?
