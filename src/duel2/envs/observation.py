"""Observation construction and normalisation.

Owner: Kyeremeh Faithful (22424515)

Layout of the 5K + 3M + 4 vector (K = queue window, M = machines):

    [0                : 5K       ]  job window, K slots x 5 features
    [5K               : 5K + 3M  ]  machine bank, M machines x 3 features
    [5K + 3M          : 5K + 3M+4]  global block

Job slot features, in order:
    processing time / p_max
    priority weight / w_max
    slack (deadline - now) / H, clipped to [-1, 1]
    waiting time (now - arrival) / H, clipped to [0, 1]
    occupancy flag in {0, 1}

Machine features, in order:
    time until free / p_max
    speed factor / s_max
    utilisation so far, already in [0, 1]

Global features, in order:
    queue length / N
    simulated time / H
    jobs not yet completed / N
    arrival rate normalised by service capacity

Every feature is relational rather than absolute: slack and waiting time are
measured against the current clock and machine state is time-until-free rather
than an absolute release time. This is what lets one network serve the whole
episode across varying congestion, and it is the state-representation
justification the rubric asks for.
"""

from __future__ import annotations

import numpy as np


def build_observation(state, cfg) -> np.ndarray:
    """Assemble the fixed-dimension observation from simulator state.

    The pending queue is sorted by (deadline, processing_time) before the first
    K jobs are read into the window, so the window always holds the K most urgent
    jobs. Unoccupied slots are zero-filled with the occupancy flag at 0.
    """
    raise NotImplementedError("TODO")


def job_slot_features(job, now: float, cfg) -> np.ndarray:
    raise NotImplementedError("TODO: 5 features, see module docstring")


def machine_features(machine, now: float, cfg) -> np.ndarray:
    raise NotImplementedError("TODO: 3 features, see module docstring")


def global_features(state, cfg) -> np.ndarray:
    raise NotImplementedError("TODO: 4 features, see module docstring")
