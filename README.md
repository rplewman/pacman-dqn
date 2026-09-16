# Ms. Pac-Man DQN

A Deep Q-Network (DQN) agent trained to play Ms. Pac-Man (`ALE/MsPacman-v5`) as a class exercise in reinforcement learning. This repo contains the executed training notebook, the code, and the evaluation evidence from one training run.

## Running it yourself

Open [pacman_dqn.ipynb](pacman_dqn.ipynb) in Jupyter, VS Code, or Google Colab with a Python 3.11-3.13 kernel. The notebook installs its own dependencies (see [requirements.txt](requirements.txt) for pinned versions) and auto-detects GPU/CPU. Edit the three hyperparameters in section 1, then Run All. The linked notebook above is the executed version, with all outputs from the run described below already visible — no need to rerun it to see the results.

## My three hyperparameters

| Setting | Value | Why |
|---|---|---|
| Exploration | **0.20** | The agent holds this epsilon constant for the entire run after a 1,000-decision random warm-up (no decay), so a lower constant value lets the learned policy actually influence the training data instead of being drowned out by persistent randomness. |
| Episodes | **25** | A step up from the notebook's suggested 5-episode sanity check, large enough to see whether any learning signal emerges, while still being a quick first run rather than a full-scale benchmark. |
| Learning rate | **0.0001** | The notebook's default and a standard, stable value for Adam-optimized DQN; the training loop raises an error on non-finite loss, which is the typical failure mode of a learning rate set too high. |

## What I expected vs. what happened

**Expected:** With only 25 episodes (about 16k decisions), I expected little or no measurable improvement — enough data to sanity-check the pipeline, not enough to learn robust Pac-Man play.

**Observed:** The trained agent's mean evaluation score more than doubled versus the untrained baseline.

| Seed | Before (untrained) | After (25 episodes) |
|---|---|---|
| 101 | 350 | 630 |
| 202 | 500 | 890 |
| 303 | 320 | 880 |
| 404 | 800 | 560 |
| 505 | 490 | 2140 |
| **Mean** | **492.0** | **1020.0** |

Change in mean score: **+528.0 (+107%)**. None of the 10 evaluation games hit the fixed time limit (all ended in a natural game over). Full breakdown: [results/comparison.json](results/comparison.json).

![Training dashboard](results/training_dashboard.png)

## Gameplay

- Before training: [results/demos/before_training.gif](results/demos/before_training.gif)

  ![Before training](results/demos/before_training.gif)

- After 25 episodes (checkpoint sample): [results/demos/episode_0025_intermediate.gif](results/demos/episode_0025_intermediate.gif)

  ![After 25 episodes](results/demos/episode_0025_intermediate.gif)

- Best of the five final evaluation games: [results/demos/final_best_trained.gif](results/demos/final_best_trained.gif)

  ![Best trained game](results/demos/final_best_trained.gif)

  This is the seed 505 game that scored **2140** overall — the highest of any evaluation game in this run. That full game actually lasted 684 decisions (~46 seconds), but the GIF cuts off after the first 300 decisions (~20 seconds), so the score visible on screen here is only a mid-game snapshot, not the final 2140. Pac-Man scoring is lumpy (chaining frightened ghosts after a power pellet gives large jumps), so a lot of that game's final score came from action after this clip ends.

**Note on all GIFs:** each one shows at most the first 20 seconds of game time (4x playback speed), but games often run longer than that before ending. The score visible in the GIF is whatever had accumulated by that 20-second cutoff — **not** the final score reported in the tables above, which always covers the entire game.

## Training budget actually used

- Completed episodes: **25 / 25**
- Total decisions: **16,346**
- Learning updates: **3,837**
- Elapsed time: **~136 seconds** (~2.3 minutes)
- Hardware: CPU only (no GPU used), Python 3.12.3, PyTorch 2.14.0+cpu, on Windows via WSL2 Ubuntu

Full config, including package versions and every fixed setting: [results/config.json](results/config.json). Per-episode log: [results/training.csv](results/training.csv). Run summary: [results/training_summary.json](results/training_summary.json). This run completed normally (not interrupted) and had learning updates from partway through episode 1 onward.

## What the agent sees, does, and is rewarded for

- **Observations:** four stacked 84x84 grayscale game screens, so the network can infer movement (a single frame only shows position).
- **Actions:** one of the joystick moves available in Ms. Pac-Man (up/down/left/right and combinations, depending on the game's action set).
- **Rewards:** the game's own points (pellets, fruit, ghosts eaten), clipped to [-1, 1] during training so no single event dominates a learning update; the scores reported above are the original, unclipped game scores.

## Limitation

Five evaluation games is a very small sample, and the mean is heavily influenced by a single standout game (seed 505 scored 2140, more than double any other game before or after training). A different set of five seeds could easily show a smaller — or no — improvement. This is a first-check result, not evidence of robust learned play.

## Next experiment

I would change **episodes only** (keeping exploration at 0.20 and learning rate at 0.0001 fixed) and scale up to around 200-300 episodes. Since epsilon does not decay in this setup, exploration and learning rate stay comparable across runs — episodes is the lever most likely to reveal whether the early improvement seen here holds up or was partly luck from a small sample.

**Update: I actually ran this, and then went further.** See below for what happened.

## Follow-up experiments: did tuning help?

After submitting the required 25-episode run above, I followed through on the "next experiment" and then kept going: a 300-episode baseline, then two additional configurations that tune settings the notebook fixes by default (replay memory size and target-network sync frequency), to see whether they could stabilize or improve results. Full methodology, results, and an honest read of what actually happened below.

### Methodology

Every run below uses the same seed (42) and the same evaluation protocol (5 fixed seeds, 5% eval exploration, 3,000-decision time limit) as the submitted run — so the "untrained" baseline scores are identical across all of them (same fixed initialization, same seed). Changes are layered on step-by-step, changing as few variables at once as possible so results stay interpretable:

1. **300 episodes** (exploration 0.20, learning rate 0.0001 unchanged) — the exact "next experiment" proposed above: does more training data alone help?
2. **300 episodes + replay memory increased from 5,000 to 100,000** — isolating just the replay buffer, to see if a longer memory (holding ~154 recent episodes instead of ~7-8) stabilizes results.
3. **300 episodes + 100k replay memory + target-network sync slowed from every 1,000 to every 2,000 decisions + exploration lowered from 0.20 to 0.10** — a combined configuration chasing the highest mean score, at the cost of not being able to attribute the result to any single change.

`REPLAY_CAPACITY` and `TARGET_EVERY` are part of the notebook's fixed classroom settings, not the three required hyperparameters — changing them is a deliberate deviation from the shared baseline, done here for extra exploration beyond the assignment's minimum ask.

Full notebooks and evidence for each: [experiments/300ep/](experiments/300ep/), [experiments/buffer100k/](experiments/buffer100k/), [experiments/combo/](experiments/combo/). Each folder contains its own config, comparison, training log/summary, dashboard, and the best-game GIF (intermediate GIFs and the untrained-baseline GIF are omitted from these follow-ups to keep the repo lean, since they're identical or available in the main [results/](results/) folder).

### Results across every run

| | Untrained | 25 episodes (submitted) | 300 episodes | 300 ep + 100k buffer | 300 ep + buffer + target-sync + explore 0.10 |
|---|---|---|---|---|---|
| Seed 101 | 350 | 630 | 620 | 410 | 380 |
| Seed 202 | 500 | 890 | 580 | 410 | 930 |
| Seed 303 | 320 | 880 | 1000 | 620 | 360 |
| Seed 404 | 800 | 560 | 430 | 570 | 430 |
| Seed 505 | 490 | **2140** | 210 | 580 | 1290 |
| **Mean** | 492.0 | **1020.0** | 568.0 | 518.0 | 678.0 |
| Episodes | 0 | 25 | 300 | 300 | 300 |
| Decisions | 0 | 16,346 | 173,600 | 178,312 | 184,848 |
| Learning updates | 0 | 3,837 | 43,151 | 44,329 | 45,963 |
| Elapsed (training loop) | — | ~136s | ~1,569s | ~1,547s | ~1,597s |

### The highest score is the 25-episode run — but read the fine print

**The submitted 25-episode run holds the highest mean score (1020.0) of everything I tried.** I'm reporting that honestly as the top result. But I don't believe it reflects the most capable agent I actually trained, and here's why: that mean is almost entirely carried by a single game (seed 505 scoring 2140 — more than double every other score in the entire table, before or after any training, in any run). Drop that one outlier and the remaining four 25-episode scores average 740, right in line with everything else here. With only 5 evaluation games, one lucky game of good ghost-avoidance can swing the whole mean. I'd call this result **lucky, not representative** — a real number, honestly earned under the fixed evaluation protocol, but not one I'd expect to reproduce reliably if I reran it with a different seed.

### What each change actually taught me

- **More episodes alone (300 vs. 25) did not help** — its mean (568) came in below the lucky 25-episode mean (1020), and well within the noise of the untrained baseline (492). Raw training scores stayed noisy throughout (see `experiments/300ep/results/training_dashboard.png`) — DQN training is not monotonic, and 300 episodes (~174k decisions) is still small next to typical Atari DQN benchmarks that use millions of frames.
- **A 100k replay buffer alone didn't raise the mean either** (518, slightly below the plain 300-episode run) — **but it dramatically tightened the spread** (410-620, vs. 210-1000 for the plain 300-episode run). A longer memory (~154 episodes vs. ~7-8) is supposed to stabilize learning, and it clearly did — it just stabilized around a modest result rather than raising the ceiling.
- **The combined change (buffer + slower target sync + lower exploration) produced the best mean of the 300-episode runs**: 678, +19% over the 300-episode baseline and +31% over buffer-only, with the improvement spread across multiple seeds (202 and 505 both improved substantially) rather than one lucky game. Since three things changed at once, I can't say which one deserves the credit — only that the combination outperformed either single change tested alone.

### Revised limitation

Beyond the original limitation above (five evaluation games is a very small, noisy sample), this round shows DQN performance here does not improve monotonically with more training or theory-backed tuning — a bigger replay buffer and a steadier target network are both textbook stability techniques, yet neither helped the mean score on its own, and only paid off once combined with each other and a lower exploration rate. Any single number in this table, including the highest one, is one noisy sample from a noisy process, not a stable measurement of the agent's skill.

### If I ran one more experiment

I'd isolate `TARGET_EVERY` alone (steadier target network, default 5,000-capacity replay buffer, default 0.20 exploration) — the one change from the combined run that was never tested by itself, so its individual contribution is still unknown.

## Notes

- Model checkpoints (`trained.pt`, `untrained.pt`, per-episode `.pt` files) are not included in this repo to keep it small; they are kept locally alongside the full run ZIP.
- `pacman_player.py` is only used for the local floating gameplay popup during interactive runs and has no effect on training or evaluation results.
