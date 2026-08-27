# Final Submission Checklist

Part C: one ZIP through Sakai LMS, named `GROUP-11_DUEL-2.zip`. Code is not submitted through
Sakai and video files are not submitted through Sakai — both are reached through the links.

```
GROUP-11_DUEL-2/
├── Project_Report.pdf
├── Hyperparameters_and_Seeds.pdf
├── AI_Use_Declaration.pdf
└── Submission_Links.txt
```

Deadline: **Friday 4 September 2026, 23:59 GMT**.

## Part D checklist

- [ ] State space, action space, reward function, termination and truncation conditions, and
      discount factor all specified
- [ ] Reward function written as an equation, not only as prose
- [ ] Addressed whether the Markov property holds under the state representation
- [ ] At least three seeds run, seed values reported
- [ ] Hyperparameters held constant across seeds and presented in a table
- [ ] Training curves plotted with mean and spread across seeds
- [ ] Evaluated on episodes not seen during training
- [ ] Exploration disabled at evaluation, and the setting stated
- [ ] Baseline run through the same code path as the agent
- [ ] A measure of variation reported for every metric
- [ ] Every figure regenerates from the submitted logs
- [ ] Exact dependency versions pinned
- [ ] README reproduces the headline result from a clean environment — **actually test this in a
      fresh virtualenv, not from memory**
- [ ] Demonstration no longer than 20 minutes
- [ ] All three members appear and present technical content
- [ ] Declaration of generative AI use included
- [ ] **Repository flipped from private to public** — `gh repo edit --visibility public --accept-visibility-change-consequences`
- [ ] GitHub repository public and opens in a signed-out browser (check it signed out; the command's
      exit status is not proof)
- [ ] Final commit SHA recorded in `Submission_Links.txt`
- [ ] YouTube link plays in a signed-out browser
- [ ] Report within 4,000 words excluding references and appendices
