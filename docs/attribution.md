# Third-Party Attribution

Rule 8: open-source libraries and published reference implementations may be used. Every such use
must be attributed **in the code at the point of use** and in the report. An unattributed
adaptation is plagiarism regardless of how much was modified.

Add a row here and a comment at the top of the adapting file for each entry.

| Source | Licence | Where used | Nature of the adaptation |
|---|---|---|---|
| CleanRL `dqn.py` (Huang et al., 2022) | MIT | `src/duel2/agent.py` | Structural reference for the DQN training loop: replay buffer, target-network sync, ε schedule, update cadence. Written out rather than imported so action masking could be threaded through exploration, greedy selection, the bootstrap target, and the dueling mean. Replay buffer extended with two mask columns. |
| Wang et al. (2016), dueling architecture | paper | `src/duel2/network.py` | The `Q = V + (A − mean A)` decomposition. Modified so the mean is taken over legal actions only. |
| Gymnasium (Towers et al., 2023) | MIT | `src/duel2/env.py` | Environment API and `check_env`. The environment itself is the group's own work. |
| PyTorch | BSD-3 | throughout | Networks and optimisation. |

Example of the in-code form:

```python
# Adapted from CleanRL dqn.py (Huang et al., 2022), MIT licence.
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
# Modified: dueling value/advantage heads; action mask applied to both the
# behaviour policy and the bootstrap target; mask stored in the replay buffer.
```
