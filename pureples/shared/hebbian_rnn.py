from neat.graphs import required_for_output
from neat.six_util import itervalues, iteritems
import numpy as np
import math
import copy


class HebbianRecurrentNetwork(object):
    def __init__(self, inputs, outputs, node_evals, firing_threshold):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals

        self.ivalues = []
        self.ovalues = []
        self.__spike_window = 3
        self.__firing_threshold = firing_threshold

        self.hebbian_buffer = {
            node: {i[0]: i[2] for i in links if i[1] > 0}
            for node, ignored_activation, ignored_aggregation, ignored_bias, ignored_response, links, learning_rate in node_evals
        }
        self.hebbian_update_log = [[copy.deepcopy(self.hebbian_buffer)]]

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
            ) in self.node_evals:
                v[node] = 0.0
                for i, w, h in links:
                    v[i] = 0.0
        self.active = 0
        # print(f"init: {self.values=}, {self.node_evals[0][5]=} {self.hebbian=}")

    def reset(self):
        # print(f"reset: {self.values=}, {self.node_evals[0][5]=} {self.hebbian=}")
        self.hebbian_buffer = {
            node: {i[0]: 0.0 for i in links if i[1] > 0}
            for node, ignored_activation, ignored_aggregation, ignored_bias, ignored_response, links, learning_rate in self.node_evals
        }
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
        # if apply:
        # print("APPLYING")
        # self.hebbian_update_log.append([copy.deepcopy(self.hebbian_buffer)])
        # print(self.hebbian_buffer)
        for idx, (
            node,
            activation,
            aggregation,
            bias,
            response,
            links,
            learning_rate,
        ) in enumerate(self.node_evals):
            # Max weight temporary hard-coded until config parsing is updated
            # if apply:
            # print(self.hebbian_buffer)
            # print(hebbians)
            link_buffer = []
            node_weight_sum = 0
            for i, w, h in links:
                # This is ugly, and could be done much cleaner
                input_val = self.ovalues[i] - self.__firing_threshold
                output_val = self.ovalues[node] - self.__firing_threshold
                # if w + response * hebbians[i] >= 0:
                if w >= 0:
                    if input_val > 0 or output_val > 0:
                        self.hebbian_buffer[node][i] = (
                            1 - learning_rate
                        ) * self.hebbian_buffer[node][i] + learning_rate * (
                            update_factor * input_val * output_val
                        )
                    if apply:
                        h = max(
                            min(
                                self.hebbian_buffer[node][i],
                                1,
                            ),
                            -1,
                        )
                        if abs(w) + h * response < 0:
                            h += (abs(w) - h * response) / response
                        self.hebbian_buffer[node][i] = h
                # node_weight_sum += w + h * response
                node_weight_sum += h * response
            if node_weight_sum > 1.0:
                self.hebbian_buffer[node] = {
                    k: v / node_weight_sum for k, v in self.hebbian_buffer[node].items()
                }

        if apply:
            self.hebbian_update_log.append([copy.deepcopy(self.hebbian_buffer)])

    def activate(self, inputs, update_hebbian=True):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError(
                "Expected {0:n} inputs, got {1:n}".format(
                    len(self.input_nodes), len(inputs)
                )
            )

        if self.ivalues and update_hebbian:
            self.update_hebbians(1, inputs[0])

        self.__prev_node_evals = copy.deepcopy(self.node_evals)

        self.ivalues = self.values[self.active]
        self.active = 1 - self.active
        self.ovalues = self.values[self.active]

        for i, v in zip(self.input_nodes, inputs):
            self.ivalues[i] = v
            self.ovalues[i] = v

        for (
            node,
            activation,
            aggregation,
            bias,
            response,
            links,
            learning_rate,
        ) in self.node_evals:
            node_inputs = [self.ivalues[i] * (w + response * h) for i, w, h in links]
            s = aggregation(node_inputs)
            self.ovalues[node] = activation(bias + s)
        return [self.ovalues[i] for i in self.output_nodes]

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
            if o not in required and i not in required:
                continue

            weight = np.sign(cg.weight) if config.binary_weights else cg.weight
            if o not in node_inputs:
                node_inputs[o] = [(i, weight, 0.0)]
            else:
                node_inputs[o].append((i, weight, 0.0))

        node_evals = []
        for node_key, inputs in iteritems(node_inputs):
            node = genome.nodes[node_key]
            activation_function = genome_config.activation_defs.get(node.activation)
            aggregation_function = genome_config.aggregation_function_defs.get(
                node.aggregation
            )
            # Temporary until config parsing is updated
            # Add variable learning rate
            node_evals.append(
                [
                    node_key,
                    activation_function,
                    aggregation_function,
                    node.bias,
                    node.response,
                    inputs,
                    config.learning_rate,
                ]
            )

        return HebbianRecurrentNetwork(
            genome_config.input_keys,
            genome_config.output_keys,
            node_evals,
            config.firing_threshold,
        )
