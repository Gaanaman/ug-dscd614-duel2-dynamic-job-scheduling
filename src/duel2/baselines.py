"""Dispatch-rule baselines and the shared policy interface.

Owner: Caleb

The agent and all three baselines implement ``Policy`` and are executed by the
same ``harness.run_policy`` call, on the same episodes, through the same metric
code. The rubric requires the baseline to run through the same code path as the
agent; a separate baseline script does not satisfy it, however carefully it is
written.

One of FCFS, SJF or Round Robin is the required baseline. All three are here
because the marginal cost is an afternoon and three reference points make the
results section far more informative than one. SJF is the honest opponent --
it is strong on average waiting time and is what a real scheduler would run, so
beating only FCFS proves less than the report will want to claim.

Each rule chooses a job. The machine is then chosen by the same tie-break for
every rule (fastest idle machine, ties by lowest index) so the comparison
isolates the job-selection rule.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    """Anything the evaluation harness can run."""

    name: str

    def reset(self) -> None:
        """Clear per-episode state. Round Robin needs this; FCFS does not."""
        ...

    def act(self, obs: np.ndarray, mask: np.ndarray, info: dict) -> int:
        """Return a valid action index.

        ``info`` carries the raw simulator view (queue and machine objects) so
        dispatch rules can read processing times directly rather than
        de-normalising them out of the observation. The learned agent uses only
        ``obs`` and ``mask``.
        """
        ...


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
