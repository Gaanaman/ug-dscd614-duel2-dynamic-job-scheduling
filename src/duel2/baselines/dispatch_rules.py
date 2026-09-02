"""FCFS, SJF and Round Robin dispatch rules.

Owner: Caleb

One of these is the required baseline; all three are implemented because the
marginal cost is an afternoon and three baselines make the comparison far more
informative than one. SJF in particular is strong on average waiting time and is
the honest opponent -- beating only FCFS proves less than the report will want
to claim.

Each rule chooses a job; the machine is then chosen by the same tie-break for
every rule (first idle machine by index, or fastest idle machine if machines are
heterogeneous) so that the comparison isolates the *job selection* rule.
"""

from __future__ import annotations

import numpy as np


class FCFS:
    """First come, first served: dispatch the earliest-arriving pending job."""

    name = "FCFS"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, mask: np.ndarray, info: dict) -> int:
        raise NotImplementedError("TODO: pick min arrival among valid slots")


class SJF:
    """Shortest job first: dispatch the pending job with the smallest p_j."""

    name = "SJF"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, mask: np.ndarray, info: dict) -> int:
        raise NotImplementedError("TODO: pick min processing_time among valid slots")


class RoundRobin:
    """Cycle through machines, taking the head of the queue each time."""

    name = "RoundRobin"

    def __init__(self, n_machines: int):
        self.n_machines = n_machines
        self._next = 0

    def reset(self) -> None:
        self._next = 0

    def act(self, obs: np.ndarray, mask: np.ndarray, info: dict) -> int:
        raise NotImplementedError(
            "TODO: advance the machine cursor to the next idle machine; take the "
            "head of the queue; fall back to no-op if nothing is dispatchable"
        )
