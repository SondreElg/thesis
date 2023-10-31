"""
This module implements a spiking neural network.
Neurons are based on the model described by:

Izhikevich, E. M.
Simple Model of Spiking Neurons
IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 14, NO. 6, NOVEMBER 2003

http://www.izhikevich.org/publications/spikes.pdf
"""

import copy
import numpy as np
from neat.attributes import FloatAttribute
from neat.genes import BaseGene, DefaultConnectionGene
from pureples.shared.genome_plus import DefaultGenomeConfig, DefaultGenome
from neat.graphs import required_for_output
from neat.six_util import itervalues

# a, b, c, d are the parameters of the Izhikevich model.
# a: the time scale of the recovery variable
# b: the sensitivity of the recovery variable
# c: the after-spike reset value of the membrane potential
# d: after-spike reset of the recovery variable
# The following parameter sets produce some known spiking behaviors:
# pylint: disable=bad-whitespace
REGULAR_SPIKING_PARAMS = {"a": 0.02, "b": 0.20, "c": -65.0, "d": 8.00}
INTRINSICALLY_BURSTING_PARAMS = {"a": 0.02, "b": 0.20, "c": -55.0, "d": 4.00}
CHATTERING_PARAMS = {"a": 0.02, "b": 0.20, "c": -50.0, "d": 2.00}
FAST_SPIKING_PARAMS = {"a": 0.10, "b": 0.20, "c": -65.0, "d": 2.00}
THALAMO_CORTICAL_PARAMS = {"a": 0.02, "b": 0.25, "c": -65.0, "d": 0.05}
RESONATOR_PARAMS = {"a": 0.10, "b": 0.25, "c": -65.0, "d": 2.00}
LOW_THRESHOLD_SPIKING_PARAMS = {"a": 0.02, "b": 0.25, "c": -65.0, "d": 2.00}


# TODO: Add mechanisms analogous to axon & dendrite propagation delay.


class IZNodeGene(BaseGene):
    """Contains attributes for the iznn node genes and determines genomic distances."""

    _gene_attributes = [
        FloatAttribute("bias"),
        FloatAttribute("a"),
        FloatAttribute("b"),
        FloatAttribute("c"),
        FloatAttribute("d"),
        FloatAttribute("response"),
    ]

    def distance(self, other, config):
        d = abs(self.bias - other.bias) + abs(self.response - other.response)
        d = (
            abs(self.a - other.a)
            + abs(self.b - other.b)
            + abs(self.c - other.c)
            + abs(self.d - other.d)
        )
        return d * config.compatibility_weight_coefficient


class IZGenome(DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        param_dict["node_gene_type"] = IZNodeGene
        param_dict["connection_gene_type"] = DefaultConnectionGene
        return DefaultGenomeConfig(param_dict)


class IZNeuron(object):
    """Sets up and simulates the iznn nodes (neurons)."""

    def __init__(self, bias, a, b, c, d, inputs, response):
        """
        a, b, c, d are the parameters of the Izhikevich model.

        :param float bias: The bias of the neuron.
        :param float a: The time-scale of the recovery variable.
        :param float b: The sensitivity of the recovery variable.
        :param float c: The after-spike reset value of the membrane potential.
        :param float d: The after-spike reset value of the recovery variable.
        :param inputs: A list of (input key, weight) pairs for incoming connections.
        :type inputs: list(tuple(int, float))
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.bias = bias
        self.inputs = inputs
        self.response = response
        self.last_fire = float("inf")

        # Membrane potential (millivolts).
        self.v = self.c

        # Membrane recovery variable.
        self.u = self.b * self.v

        self.fired = 0.0
        self.current = self.bias

    def advance(self, dt_msec):
        """
        Advances simulation time by the given time step in milliseconds.

        v' = 0.04 * v^2 + 5v + 140 - u + I
        u' = a * (b * v - u)

        if v >= 30 then
            v <- c, u <- u + d
        """
        # TODO: Make the time step adjustable, and choose an appropriate
        # numerical integration method to maintain stability.
        # TODO: The need to catch overflows indicates that the current method is
        # not stable for all possible network configurations and states.
        try:
            self.v += (
                0.5
                * dt_msec
                * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + self.current)
            )
            self.v += (
                0.5
                * dt_msec
                * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + self.current)
            )
            self.u += dt_msec * self.a * (self.b * self.v - self.u)
        except OverflowError:
            # Reset without producing a spike.
            self.v = self.c
            self.u = self.b * self.v

        self.fired = 0.0
        self.last_fire += 1
        if self.v > 30.0:
            # Output spike and reset.
            self.fired = 1.0
            self.last_fire = 0
            self.v = self.c
            self.u += self.d

    def reset(self):
        """Resets all state variables."""
        self.v = self.c
        self.u = self.b * self.v
        self.fired = 0.0
        self.last_fire = float("inf")
        self.current = self.bias


class IZNN(object):
    """Basic iznn network object."""

    def __init__(self, neurons, inputs, outputs):
        self.neurons = neurons
        self.inputs = inputs
        self.outputs = outputs
        self.input_values = {}
        self.hebbian_buffer = {
            index: {i[0]: i[2] for i in neuron.inputs}
            for index, neuron in neurons.items()
        }
        self.hebbian_update_log = [[copy.deepcopy(self.hebbian_buffer)]]
        self.apply_hebbian = 0

    def set_inputs(self, inputs):
        """Assign input voltages."""
        if len(inputs) != len(self.inputs):
            raise RuntimeError(
                "Number of inputs {0:d} does not match number of input nodes {1:d}".format(
                    len(inputs), len(self.inputs)
                )
            )
        for i, v in zip(self.inputs, inputs):
            self.input_values[i] = v

    def reset(self):
        """Reset all neurons to their default state."""
        for n in itervalues(self.neurons):
            n.reset()

    def get_time_step_msec(self):
        # pylint: disable=no-self-use
        # TODO: Investigate performance or numerical stability issues that may
        # result from using this hard-coded time step.
        return 0.05

    # update hebbians at each timestep where they fire
    def update_hebbians(self, dt_msec, apply, A_plus=0.01, A_minus=0.011, tau_msec=20):
        for index, n in self.neurons.items():
            for connection_index, (input_neuron_index, w, h) in enumerate(n.inputs):
                if w < 0:
                    continue  # Not implemented for inhibitory connections
                ineuron = self.neurons.get(input_neuron_index)
                if ineuron is None:
                    continue  # Need to track input neuron firing rate?

                time_diff = ineuron.last_fire * dt_msec - n.last_fire * dt_msec
                if n.last_fire == 0 and ineuron.last_fire * dt_msec < tau_msec:
                    # update for post
                    # print(
                    #     f"UPDATE POSITIVE {self.hebbian_buffer[index][input_neuron_index]}"
                    # )
                    self.hebbian_buffer[index][input_neuron_index] += A_plus * np.exp(
                        time_diff / tau_msec
                    )
                elif ineuron.last_fire == 0 and n.last_fire * dt_msec < tau_msec:
                    # update for pre
                    # print(
                    #     f"UPDATE NEGATIVE {self.hebbian_buffer[index][input_neuron_index]}"
                    # )
                    self.hebbian_buffer[index][input_neuron_index] -= A_minus * np.exp(
                        -time_diff / tau_msec
                    )
                if apply:
                    # print(f"APPLY {self.hebbian_buffer[index][input_neuron_index]}")
                    # print(self.neurons[index].inputs)
                    # print(h)
                    h = max(
                        min(
                            self.hebbian_buffer[index][input_neuron_index],
                            1,
                        ),
                        -1,
                    )
                    if abs(w) + h * n.response < 0:
                        h = (h * n.response - (abs(w) + h * n.response)) / n.response
                    self.neurons[index].inputs[connection_index] = (
                        input_neuron_index,
                        w,
                        h,
                    )
                    self.hebbian_buffer[index][input_neuron_index] = h
                    # print(self.neurons[index].inputs)
                    # print(h)
        if apply:
            self.hebbian_update_log.append(copy.deepcopy([self.hebbian_buffer]))

    def advance(self, dt_msec):
        for n in itervalues(self.neurons):
            n.current = n.bias
            for i, w, h in n.inputs:
                ineuron = self.neurons.get(i)
                if ineuron is not None:
                    ivalue = ineuron.fired
                else:
                    ivalue = self.input_values[i]

                n.current += ivalue * (w + h * n.response)

        for n in itervalues(self.neurons):
            n.advance(dt_msec)

        # print(self.input_values)
        if self.input_values[-1] == 0:
            self.apply_hebbian = 0
        else:
            self.apply_hebbian += int(self.input_values[-1] == 1)
        self.update_hebbians(dt_msec, self.apply_hebbian == 1)

        return [self.neurons[i].fired for i in self.outputs]

    @staticmethod
    def create(genome, config):
        """Receives a genome and returns its phenotype (a neural network)."""
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

            if o not in node_inputs:
                node_inputs[o] = [(i, cg.weight, 0)]
            else:
                node_inputs[o].append((i, cg.weight, 0))

        neurons = {}
        for node_key in required:
            ng = genome.nodes[node_key]
            inputs = node_inputs.get(node_key, [])
            neurons[node_key] = IZNeuron(
                ng.bias, ng.a, ng.b, ng.c, ng.d, inputs, ng.response
            )

        genome_config = config.genome_config
        return IZNN(neurons, genome_config.input_keys, genome_config.output_keys)
