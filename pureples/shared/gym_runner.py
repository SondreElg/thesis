"""
Generic runner for AI Gym - runs Neat, Hyperneat and ES-Hyperneat
"""

import neat
import numpy as np
import multiprocessing
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.es_hyperneat.es_hyperneat import ESNetwork
from pureples.es_hyperneat_rnn.es_hyperneat_rnn import ESNetworkRNN
from pureples.shared.concurrent_neat_population import Population


def ini_pop(state, stats, config, output):
    """
    Initialize population attaching statistics reporter.
    """
    pop = neat.population.Population(config, state)
    if output:
        pop.add_reporter(neat.reporting.StdOutReporter(True))
    pop.add_reporter(stats)
    return pop


def conc_ini_pop(state, stats, config, output):
    """
    Initialize population attaching statistics reporter.
    """
    pop = Population(config, state)
    if output:
        pop.add_reporter(neat.reporting.StdOutReporter(True))
    pop.add_reporter(stats)
    return pop


def run_es_snn():
    return


def _eval_fitness(genome, config):
    cppn = neat.nn.FeedForwardNetwork.create(genome, config)
    network = ESNetworkRNN(config.substrate, cppn, config.params)
    net = network.create_phenotype_network()

    fitnesses = []
    for _ in range(config.trials):
        ob = config.env.reset()[0].tolist()
        net.reset()

        total_reward = 0

        for _ in range(config.max_steps):
            for _ in range(network.activations):
                o = net.activate(ob)

            action = np.argmax(o)
            ob, reward, done, _ = [*(config.env.step(action))][:-1]
            total_reward += reward
            if done:
                break
        fitnesses.append(total_reward)

    # genome fitness
    return np.array(fitnesses).mean()


def run_es_rnn(
    gens,
    env,
    max_steps,
    config,
    params,
    substrate,
    max_trials=100,
    output=True,
    processes=1,
):
    """
    Generic OpenAI Gym runner for ES-HyperNEAT-RNN.
    """
    trials = 1

    def eval_fitness(genomes, config):
        for _, g in genomes:
            g.fitness = _eval_fitness(genomes, config)

    setattr(config, "env", env)
    setattr(config, "max_steps", max_steps)
    setattr(config, "params", params)
    setattr(config, "substrate", substrate)
    setattr(config, "trials", trials)

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), _eval_fitness)
    pop.run(pe.evaluate, gens)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10
    winner_ten = pop.run(pe.evaluate, gens)

    if max_trials == 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(pe.evaluate, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)


def run_es(
    gens,
    env,
    max_steps,
    config,
    params,
    substrate,
    max_trials=100,
    output=True,
):
    """
    Generic OpenAI Gym runner for ES-HyperNEAT.
    """
    trials = 1

    def eval_fitness(genomes, config):
        for _, g in genomes:
            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            network = ESNetwork(substrate, cppn, params)
            net = network.create_phenotype_network()

            fitnesses = []

            for _ in range(trials):
                ob = env.reset()[0].tolist()
                net.reset()

                total_reward = 0

                for _ in range(max_steps):
                    for _ in range(network.activations):
                        o = net.activate(ob)

                    action = np.argmax(o)
                    ob, reward, done, _ = [*(env.step(action))][:-1]
                    total_reward += reward
                    if done:
                        break

                fitnesses.append(total_reward)

            g.fitness = np.array(fitnesses).mean()

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)
    pop.run(eval_fitness, gens)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10
    winner_ten = pop.run(eval_fitness, gens)

    if max_trials == 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(eval_fitness, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)


def run_hyper(
    gens,
    env,
    max_steps,
    config,
    substrate,
    activations,
    max_trials=100,
    activation="sigmoid",
    output=True,
):
    """
    Generic OpenAI Gym runner for HyperNEAT.
    """
    trials = 1

    def eval_fitness(genomes, config):
        for _, g in genomes:
            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            net = create_phenotype_network(cppn, substrate, activation)

            fitnesses = []

            for _ in range(trials):
                ob = env.reset()
                net.reset()

                total_reward = 0

                for _ in range(max_steps):
                    for _ in range(activations):
                        o = net.activate(ob)
                    action = np.argmax(o)
                    ob, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
                fitnesses.append(total_reward)

            g.fitness = np.array(fitnesses).mean()

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)
    pop.run(eval_fitness, gens)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10
    winner_ten = pop.run(eval_fitness, gens)

    if max_trials == 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(eval_fitness, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)


def run_neat(gens, env, max_steps, config, max_trials=100, output=True):
    """
    Generic OpenAI Gym runner for NEAT.
    """
    trials = 1

    def eval_fitness(genomes, config):
        for _, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)

            fitnesses = []

            for _ in range(trials):
                ob = env.reset()

                total_reward = 0

                for _ in range(max_steps):
                    o = net.activate(ob)
                    action = np.argmax(o)
                    ob, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
                fitnesses.append(total_reward)

            g.fitness = np.array(fitnesses).mean()

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)
    pop.run(eval_fitness, gens)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10
    winner_ten = pop.run(eval_fitness, gens)

    if max_trials == 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(eval_fitness, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)
