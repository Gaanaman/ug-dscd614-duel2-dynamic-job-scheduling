# Third-Party Attribution

Rule 8: open-source libraries and published reference implementations may be used. Every such use
must be attributed **in the code at the point of use** and in the report. An unattributed
adaptation is plagiarism regardless of how much was modified.

Add a row here and a comment at the top of the adapting file for each entry.

| Source | Licence | Where used | Nature of the adaptation |
|---|---|---|---|
| | | | |

Example of the in-code form:

```python
# Adapted from CleanRL dqn.py (Huang et al., 2022), MIT licence.
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
# Modified: dueling value/advantage heads; action mask applied to both the
# behaviour policy and the bootstrap target; mask stored in the replay buffer.
```
