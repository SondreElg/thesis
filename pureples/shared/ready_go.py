import copy
import math
import numpy as np


def ready_go_list(
    foreperiod,
    cycles,
    time_block_size=None,
    cycle_delay_range=[0, 6],
    distributions=[],
    distributions_args=[{}],
    trials=1,
):
    """Returns a list of inputs representing the ready-go sequence with delays between each cycle"""
    generation_data = []

    cycle_len = math.floor(foreperiod / time_block_size)
    for i in range(trials * len(distributions)):
        inputs = []
        dist_index = i % len(distributions)
        for _ in range(cycles):
            cycle_spike = int(
                max(
                    min(
                        distributions[dist_index](**distributions_args[dist_index]),
                        cycle_len - 1,
                    ),
                    0,
                )
            )
            inputs += np.hstack(
                [1]
                + [2 if i == cycle_spike else 0 for i in range(cycle_len)]
                + np.zeros(
                    np.random.randint(cycle_delay_range[0], cycle_delay_range[1])
                ).tolist()
            ).tolist()

        expected_output = []
        ready = 0
        state = 0
        distance = 999
        for index, input in enumerate(inputs):
            go = int(input == 2)
            state = 2 if go else 1 if (ready or state == 1) else 0
            ready = int(input == 1)

            # if ready:
            #     predicted = False
            # elif state == 2:
            #     predicted = True

            if state == 2:
                expected_output.append(1)
            else:
                expected_output.append(0)
            # if state == 1:
            #     distance = inputs.index(2, index) - index + 1
            # else:
            #     if state == 2:
            #         # expected_output.append(1)
            #         distance = 0
            #     # else:
            #     #     expected_output.append(0)
            #     distance += 1
            # expected_output.append(1 / distance**2)

        generation_data.append([inputs, expected_output])
    # order = [0, 4, 2, 3, 1]
    generation_data = np.array(generation_data, dtype=object)
    shuffled_data = copy.deepcopy(generation_data)
    flipped_data = np.flip(generation_data, axis=0)
    while np.array_equal(shuffled_data, generation_data) or np.array_equal(
        shuffled_data, flipped_data
    ):
        np.random.shuffle(shuffled_data)
    return shuffled_data


def ready_go_list_zip(
    foreperiod,
    cycles,
    time_block_size=None,
    cycle_delay_range=[0, 1],
    distributions=[],
    distributions_args=[{}],
    trials=1,
):
    """Returns a list of inputs representing the ready-go sequence with delays between each cycle"""
    generation_data = []

    cycle_len = math.floor(foreperiod / time_block_size)
    for _ in range(trials):
        inputs = []
        for j in range(cycles):
            dist_index = j % len(distributions)
            cycle_spike = int(
                max(
                    min(
                        distributions[dist_index](**distributions_args[dist_index]),
                        cycle_len - 1,
                    ),
                    0,
                )
            )
            inputs += np.hstack(
                [1]
                + [2 if i == cycle_spike else 0 for i in range(cycle_len)]
                + np.zeros(
                    np.random.randint(cycle_delay_range[0], cycle_delay_range[1])
                ).tolist()
            ).tolist()

        expected_output = []
        ready = 0
        state = 0
        distance = 999
        for index, input in enumerate(inputs):
            go = int(input == 2)
            state = 2 if go else 1 if (ready or state == 1) else 0
            ready = int(input == 1)
            # net.reset()

            # if ready:
            #     predicted = False
            # elif state == 2:
            #     predicted = True

            # Why does it activate multiple times?
            # Does the recurrent network implementation progress one "layer" at a time?

            if state == 2:
                expected_output.append(1)
            else:
                expected_output.append(0)
            # if state == 1:
            #     distance = inputs.index(2, index) - index + 1
            # else:
            #     if state == 2:
            #         # expected_output.append(1)
            #         distance = 0
            #     # else:
            #     #     expected_output.append(0)
            #     distance += 1
            # expected_output.append(1 / distance**2)

        generation_data.append([inputs, expected_output])
    return generation_data


def old_ready_go_list(
    foreperiod, cycles, time_block_size=None, cycle_delay_range=[0, 1], trials=1
):
    """Returns a list of inputs representing the ready-go sequence with delays between each cycle"""

    # if time_block_size:
    #
    # else:
    #     # Do not discretizise output
    #     return
    generation_data = []

    for _ in range(trials):
        cycle_len = math.floor(np.random.choice(foreperiod) / time_block_size)
        inputs = np.hstack(
            [
                [1]
                + [2 if i == cycle_len - 1 else 0 for i in range(cycle_len)]
                + np.zeros(
                    np.random.randint(cycle_delay_range[0], cycle_delay_range[1])
                ).tolist()
                for _ in range(cycles)
            ]
        ).tolist()

        expected_output = []
        ready = 0
        state = 0
        distance = 999
        for index, input in enumerate(inputs):
            go = int(input == 2)
            state = 2 if go else 1 if (ready or state == 1) else 0
            ready = int(input == 1)
            # net.reset()

            # if ready:
            #     predicted = False
            # elif state == 2:
            #     predicted = True

            # Why does it activate multiple times?
            # Does the recurrent network implementation progress one "layer" at a time?

            if state == 1:
                distance = inputs.index(2, index) - index + 1
            else:
                if state == 2:
                    distance = 0
                distance += 1
            expected_output.append(1 / distance**2)

        generation_data.append([inputs, expected_output])
    return generation_data
