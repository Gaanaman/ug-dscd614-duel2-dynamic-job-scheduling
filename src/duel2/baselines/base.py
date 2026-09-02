"""Common policy interface.

Owner: Caleb

The agent and all three baselines implement this interface and are executed by
the same ``evaluation.harness.run_policy`` call on the same episodes with the
same metric code. The rubric requires the baseline to run through the same code
path as the agent; a separate baseline script does not satisfy it, however
carefully it is written.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    """Anything the evaluation harness can run."""

    name: str

    def reset(self) -> None:
        """Clear any per-episode state. Round Robin needs this; FCFS does not."""
        ...

    def act(self, obs: np.ndarray, mask: np.ndarray, info: dict) -> int:
        """Return a valid action index.

        ``info`` carries the raw simulator view (queue and machine objects) so
        that dispatch-rule baselines can read processing times directly rather
        than de-normalising them out of the observation vector. The learned
        agent uses only ``obs`` and ``mask``.
        """
        ...
