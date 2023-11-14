import copy
import math
import numpy as np


def shuffle_data(generation_data, flip_pad=False, ordering=[]):
    if ordering:
        shuffled = ordering
    else:
        ordered = np.arange(len(generation_data))
        shuffled = np.arange(len(generation_data))
        flipped = np.flip(ordered)
        while np.array_equal(shuffled, ordered) or np.array_equal(shuffled, flipped):
            np.random.shuffle(shuffled)

    shuffled_data = generation_data[shuffled]
    if flip_pad:
        shuffled_data = np.append(
            shuffled_data, np.flip(shuffled_data[:-1], axis=0), axis=0
        )
    return shuffled_data


def foreperiod_rg_list(
    foreperiod,
    cycles,
    time_block_size=None,
    cycle_delay_range=[0, 6],
    foreperiods=[],
    flip_pad=False,
    ordering=[],
    repetitions=1,
):
    """Returns a list of inputs representing the ready-go sequence with delays between each cycle"""
    generation_data = np.empty((repetitions * len(foreperiods), 2), dtype=object)

    cycle_len = math.floor(foreperiod / time_block_size)
    for i in range(repetitions * len(foreperiods)):
        inputs = []
        fp = foreperiods[i % len(foreperiods)]
        repeated = np.zeros(
            1 + cycle_len,
            dtype=int,
        )
        repeated[0] = 1  # Mark the start of the trial with a 1
        repeated[fp] = 2
        trials = []
        for _ in range(cycles):
            trial = np.append(
                repeated,
                np.zeros(np.random.randint(cycle_delay_range[0], cycle_delay_range[1])),
            )
            trials.append(trial)

        inputs = np.hstack(trials)
        expected_output = np.zeros_like(inputs, dtype=int)
        expected_output[np.where(inputs == 2)] = 1

        generation_data[i] = [inputs, expected_output]

    return shuffle_data(generation_data, flip_pad, ordering)


def ready_go_list(
    foreperiod,
    cycles,
    time_block_size=None,
    cycle_delay_range=[0, 6],
    distributions=[],
    distributions_args=[{}],
    flip_pad=False,
    repetitions=1,
):
    """Returns a list of inputs representing the ready-go sequence with delays between each cycle"""
    generation_data = np.empty((repetitions * len(distributions), 2), dtype=object)

    cycle_len = math.floor(foreperiod / time_block_size)
    for i in range(repetitions * len(distributions)):
        inputs = []
        dist_index = i % len(distributions)
        trials = []
        for _ in range(cycles):
            trial = np.zeros(
                1
                + cycle_len
                + np.random.randint(cycle_delay_range[0], cycle_delay_range[1]),
                dtype=int,
            )
            trial[0] = 1  # Mark the start of the trial with a 1

            # Set a random position within the trial_len as 2, excluding the first position
            cycle_spike = int(
                max(
                    min(
                        distributions[dist_index](**distributions_args[dist_index]),
                        cycle_len,
                    ),
                    1,
                )
            )
            trial[cycle_spike] = 2
            trials.append(trial)

        # Concatenate all trials to form the final array
        inputs = np.hstack(trials)

        expected_output = np.zeros_like(inputs, dtype=int)
        expected_output[np.where(inputs == 2)] = 1

        generation_data[i] = [inputs, expected_output]
        # generation_data[i] = [inputs.tolist(), expected_output.tolist()]

    return shuffle_data(generation_data, flip_pad)


def ready_go_list_zip(
    foreperiod,
    cycles,
    time_block_size=None,
    cycle_delay_range=[0, 1],
    distributions=[],
    distributions_args=[{}],
    repetitions=1,
):
    """Returns a list of inputs representing the ready-go sequence with delays between each cycle"""
    generation_data = []

    cycle_len = math.floor(foreperiod / time_block_size)
    for _ in range(repetitions):
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
    foreperiod, cycles, time_block_size=None, cycle_delay_range=[0, 1], repetitions=1
):
    """Returns a list of inputs representing the ready-go sequence with delays between each cycle"""

    # if time_block_size:
    #
    # else:
    #     # Do not discretizise output
    #     return
    generation_data = []

    for _ in range(repetitions):
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
