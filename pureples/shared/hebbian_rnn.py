from neat.graphs import required_for_output
from neat.six_util import itervalues, iteritems
import numpy as np
import math
import copy


class HebbianRecurrentNetwork(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals

        self.__ivalues = []
        self.__ovalues = []
        self.__spike_window = 3
        self.hebbian_buffer = [
            dict([(i, 0.0) for i in eval[7].keys()]) for eval in node_evals
        ]
        self.hebbian_update_log = []

        self.values = [{}, {}]
        for v in self.values:
            for k in inputs + outputs:
                v[k] = 0.0

            for (
                node,
                ignored_activation,
                ignored_aggregation,
                ignored_bias,
                ignored_response,
                links,
                learning_rate,
                hebbians,
            ) in self.node_evals:
                v[node] = 0.0
                for i, w in links:
                    v[i] = 0.0
        self.active = 0
        # print(f"init: {self.values=}, {self.node_evals[0][5]=} {self.hebbian=}")

    def reset(self):
        # print(f"reset: {self.values=}, {self.node_evals[0][5]=} {self.hebbian=}")
        self.hebbian_buffer = []
        for i, (
            node,
            ignored_activation,
            ignored_aggregation,
            ignored_bias,
            ignored_response,
            links,
            learning_rate,
            hebbians,
        ) in enumerate(self.node_evals):
            hebbians = dict([(i, 0.0) for i in hebbians.keys()])
            self.hebbian_buffer.append(dict([(i, 0.0) for i in hebbians.keys()]))
            self.hebbian_update_log = []
            self.node_evals[i] = [
                node,
                ignored_activation,
                ignored_aggregation,
                ignored_bias,
                ignored_response,
                links,
                learning_rate,
                hebbians,
            ]
        # self.hebbian_buffer = [eval[7] for eval in self.node_evals]
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.active = 0

    def soft_reset(self):
        # print(f"soft_reset: {self.values=}, {self.node_evals[0][5]=} {self.hebbian=}")
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.active = 0

    def update_hebbians(self, update_factor, apply):
        # Arbitrarily chosen limit for what is considered "good" fitness
        # if fitness < 0.70:
        #     return
        # print("WE HERE BOIS")
        if apply:
            # print("APPLYING")
            self.hebbian_update_log.append(
                copy.deepcopy([eval[7] for eval in self.node_evals])
            )
        # print(self.hebbian_buffer)
        for idx, (
            node,
            activation,
            aggregation,
            bias,
            response,
            links,
            learning_rate,
            hebbians,
        ) in enumerate(self.__prev_node_evals):
            # Max weight temporary hard-coded until config parsing is updated
            # if apply:
            # print(self.hebbian_buffer)
            # print(hebbians)
            firing_threshold = 0.20
            for i, w in links:
                # This is ugly, and could be done much cleaner
                input_val = self.__ivalues[i]
                output_val = self.__ovalues[i]
                # if w + response * hebbians[i] >= 0:
                if w >= 0:
                    if input_val > firing_threshold and output_val > firing_threshold:
                        self.hebbian_buffer[idx][i] = (
                            1 - learning_rate
                        ) * self.hebbian_buffer[idx][i] + learning_rate * (
                            update_factor * input_val * output_val - firing_threshold
                        )
                    elif input_val > firing_threshold or output_val > firing_threshold:
                        self.hebbian_buffer[idx][i] = (
                            1 - learning_rate
                        ) * self.hebbian_buffer[idx][i] - learning_rate * (
                            update_factor * input_val * output_val - firing_threshold
                        )
                    # else:
                    #     self.hebbian_buffer[idx][i] = (
                    #         1 - learning_rate
                    #     ) * self.hebbian_buffer[idx][i] + learning_rate * (
                    #         update_factor * input_val * output_val - firing_threshold
                    #     )
                    if apply:
                        hebbians[i] = max(
                            min(
                                hebbians[i] + self.hebbian_buffer[idx][i],
                                10,
                            ),
                            0,
                        )
                # if w + response * hebbians[i] < 0:
                if w < 0:
                    if input_val > firing_threshold and output_val > firing_threshold:
                        self.hebbian_buffer[idx][i] = (
                            1 - learning_rate
                        ) * self.hebbian_buffer[idx][i] + learning_rate * (
                            update_factor * input_val * output_val - firing_threshold
                        )
                    elif input_val > firing_threshold or output_val > firing_threshold:
                        self.hebbian_buffer[idx][i] = (
                            1 - learning_rate
                        ) * self.hebbian_buffer[idx][i] - learning_rate * (
                            update_factor * input_val * output_val - firing_threshold
                        )
                    # else:
                    #     self.hebbian_buffer[idx][i] = (
                    #         1 - learning_rate
                    #     ) * self.hebbian_buffer[idx][i] + learning_rate * (
                    #         update_factor * input_val * output_val - firing_threshold
                    #     )
                    if apply:
                        hebbians[i] = min(
                            max(
                                hebbians[i] + self.hebbian_buffer[idx][i],
                                -10,
                            ),
                            0,
                        )
                if apply:
                    # hebbians[i] = (
                    #     hebbians[i] + self.hebbian_buffer[idx][i] * learning_rate
                    # )
                    # print(hebbians[i])
                    self.hebbian_buffer[idx][i] = 0
                    # print(hebbians[i])

            if apply:
                # print(self.hebbian_buffer[idx])
                # print(hebbians)
                self.node_evals[idx] = self.node_evals[idx][:-1] + [hebbians]
                # print(self.node_evals[idx])

    def activate(self, inputs, prev_fitness):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError(
                "Expected {0:n} inputs, got {1:n}".format(
                    len(self.input_nodes), len(inputs)
                )
            )

        if self.__ivalues:
            self.update_hebbians(inputs[1], inputs[0] == 1)

        self.__prev_node_evals = copy.deepcopy(self.node_evals)

        self.__ivalues = self.values[self.active]
        self.active = 1 - self.active
        self.__ovalues = self.values[self.active]

        for i, v in zip(self.input_nodes, inputs):
            self.__ivalues[i] = v
            self.__ovalues[i] = v

        for (
            node,
            activation,
            aggregation,
            bias,
            response,
            links,
            learning_rate,
            hebbians,
        ) in self.node_evals:
            node_inputs = [
                self.__ivalues[i] * (w + response * hebbians[i]) for i, w in links
            ]
            s = aggregation(node_inputs)
            self.__ovalues[node] = activation(bias + s)
        return [self.__ovalues[i] for i in self.output_nodes]

    @staticmethod
    def create(genome, config):
        """Receives a genome and returns its phenotype (a RecurrentNetwork)."""
        genome_config = config.genome_config
        required = required_for_output(
            genome_config.input_keys, genome_config.output_keys, genome.connections
        )

        # Gather inputs and expressed connections.
        node_inputs = {}
        for cg in itervalues(genome.connections):
            if not cg.enabled:
                continue

            i, o = cg.key
            # Last part technically breaks the no-direct clause, but may be kept depending on performance
            if o not in required and i not in required or (i < 0 and o == 0):
                continue

            if o not in node_inputs:
                node_inputs[o] = [(i, cg.weight)]
            else:
                node_inputs[o].append((i, cg.weight))

        node_evals = []
        for node_key, inputs in iteritems(node_inputs):
            node = genome.nodes[node_key]
            activation_function = genome_config.activation_defs.get(node.activation)
            aggregation_function = genome_config.aggregation_function_defs.get(
                node.aggregation
            )
            # Temporary until config parsing is updated
            # Add variable learning rate and embed hebbian values to node_evals
            learning_rate = 0.05
            hebbians = dict([(i, 0.0) for i, _ in inputs])
            node_evals.append(
                [
                    node_key,
                    activation_function,
                    aggregation_function,
                    node.bias,
                    node.response,
                    inputs,
                    learning_rate,
                    hebbians,
                ]
            )

        return HebbianRecurrentNetwork(
            genome_config.input_keys, genome_config.output_keys, node_evals
        )
