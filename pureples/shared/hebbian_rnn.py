from neat.graphs import required_for_output
from neat.six_util import itervalues, iteritems
import numpy as np
import math


class HebbianRecurrentNetwork(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals

        self.__ivalues = []
        self.__ovalues = []
        self.__spike_window = 3
        self.hebbian_buffer = node_evals[7]

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
        # reset_hebbians = []
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
            self.node_evals[i] = (
                node,
                ignored_activation,
                ignored_aggregation,
                ignored_bias,
                ignored_response,
                links,
                learning_rate,
                hebbians,
            )
        # if self.node_evals:
        #     self.hebbian = dict(
        #         [(i, 0.0) for i, _ in [entry[5] for entry in self.node_evals][0]]
        #     )
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.active = 0

    def soft_reset(self):
        # print(f"soft_reset: {self.values=}, {self.node_evals[0][5]=} {self.hebbian=}")
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.active = 0

    def update_hebbians(self, fitness):
        # Arbitrarily chosen limit for what is considered "good" fitness
        # if fitness < 0.70:
        #     return
        for idx, (
            node,
            activation,
            aggregation,
            bias,
            response,
            links,
            learning_rate,
            hebbians,
        ) in enumerate(self.node_evals):
            # Max weight temporary hard-coded until config parsing is updated
            for i, w in links:
                # This is ugly, and could be done much cleaner
                input_val = self.__ivalues[i] * abs(w + hebbians[i])
                output_val = self.__ovalues[i] * abs(w + hebbians[i])
                if w > 0:
                    if input_val > 0.15 and output_val > 0.15:
                        hebbians[i] = (
                            min(
                                w
                                + hebbians[i]
                                + (w + hebbians[i]) * input_val * output_val,
                                30,
                            )
                            - w
                        )
                    # elif input_val > 0.15 and output_val < 0.15:
                    #     hebbians[i] = max(w + hebbians[i] - learning_rate, 0.05) - w
                    elif input_val > 0.15 or output_val > 0.15:
                        hebbians[i] = (
                            max(
                                w
                                + hebbians[i]
                                - (w + hebbians[i]) * input_val * output_val,
                                0.05,
                            )
                            - w
                        )
                if w < 0:
                    if input_val > 0.15 and output_val > 0.15:
                        hebbians[i] = (
                            min(
                                w
                                + hebbians[i]
                                + (w + hebbians[i]) * input_val * output_val,
                                -0.5,
                            )
                            - w
                        )
                    elif input_val > 0.15 and output_val < 0.15:
                        hebbians[i] = (
                            max(
                                w
                                + hebbians[i]
                                - (w + hebbians[i]) * input_val * output_val,
                                -30,
                            )
                            - w
                        )
                # elif input_val < 0.15 and output_val < 0.15:
                #     hebbians[i] = max(w + hebbians[i] - learning_rate, -30) - w
            self.node_evals[idx] = (
                node,
                activation,
                aggregation,
                bias,
                response,
                links,
                learning_rate,
                hebbians,
            )

    def activate(self, inputs, prev_fitness):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError(
                "Expected {0:n} inputs, got {1:n}".format(
                    len(self.input_nodes), len(inputs)
                )
            )

        if self.__ivalues:
            self.update_hebbians(prev_fitness)

        self.__prev_node_evals = self.node_evals

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
            node_inputs = [self.__ivalues[i] * (w + hebbians[i]) for i, w in links]
            s = aggregation(node_inputs)
            self.__ovalues[node] = activation(bias + response * s)
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
            if o not in required and i not in required or i + o < 0:
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
                (
                    node_key,
                    activation_function,
                    aggregation_function,
                    node.bias,
                    node.response,
                    inputs,
                    learning_rate,
                    hebbians,
                )
            )

        return HebbianRecurrentNetwork(
            genome_config.input_keys, genome_config.output_keys, node_evals
        )
