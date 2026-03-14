# moral_harvest

PPO in Melting Pot Harvest with reward shaping.

## Single-agent PPO (Step 1)

The current runnable path is a **single PPO focal agent** in `commons_harvest__open`.
The wrapper keeps only local observations (`RGB` by default) and assigns no-op actions to all non-focal agents.

Both backends are available:
- `cleanrl` (default): CleanRL PPO implementation.
- `rllib`: RLlib PPO implementation.

### Run from WSL using `.venv-linux`

From Windows PowerShell:

```powershell
wsl
```

```bash
cd /mnt/c/Users/dchen/Documents/Projects/moral_harvest && .venv-linux/bin/python -m moral_harvest.cli.train --mode single-agent --substrate commons_harvest__open --focal-agent player_0 --stop-iters 300 --checkpoint-every 100 --checkpoint-root checkpoints/single_agent/cleanrl
```

### Run CleanRL backend from WSL using `.venv-linux`

From Windows PowerShell:

```powershell
wsl
```

```bash
cd /mnt/c/Users/dchen/Documents/Projects/moral_harvest && .venv-linux/bin/python -m moral_harvest.cli.train --mode single-agent --backend cleanrl --substrate commons_harvest__open --focal-agent player_0 --stop-iters 300 --checkpoint-every 100 --checkpoint-root checkpoints/single_agent/cleanrl
```

Notes:
- Checkpoints are saved every `--checkpoint-every` iterations and once at the end.
- Training logs print iteration reward and learner stats (`policy_loss`, `value_loss`, `entropy`).
- Per-iteration metrics are saved under `results/single_agent/<backend>/<run_name>/` as both `metrics.jsonl` and `metrics.csv`.

### Evaluate a saved single-agent checkpoint

Replace `CHECKPOINT_PATH` below with a path printed by training.
- RLlib checkpoint: a directory path.
- CleanRL checkpoint: a `.pt` file path.

```powershell
wsl
```
```bash
cd /mnt/c/Users/dchen/Documents/Projects/moral_harvest && .venv-linux/bin/python -m moral_harvest.cli.rollout_single_agent --backend auto --checkpoint-path checkpoints/single_agent/rllib/<your_checkpoint_dir> --episodes 5 --substrate commons_harvest__open --focal-agent player_0
```

### Evaluate + record video

Use the same rollout script with `--record-video`. Videos are written under `videos/<run_name>/`.

```powershell
wsl
```

```bash
cd /mnt/c/Users/dchen/Documents/Projects/moral_harvest && .venv-linux/bin/python -m moral_harvest.cli.rollout_single_agent --backend auto --checkpoint-path checkpoints/single_agent/rllib/<your_checkpoint_dir> --episodes 5 --substrate commons_harvest__open --focal-agent player_0 --record-video --run-name single_agent_eval
```

Notes:
- `--explore` enables stochastic action sampling during rollout.
- `--seed` sets the rollout seed for reproducibility.

### Plot training curves from saved metrics

After training, use the plotting module on either `metrics.jsonl` or `metrics.csv`.

```powershell
wsl
```

```bash
cd /mnt/c/Users/dchen/Documents/Projects/moral_harvest && .venv-linux/bin/python -m moral_harvest.analysis.plot_training_curves --metrics-path results/single_agent/rllib/<run_name>/metrics.jsonl --y-keys episode_reward_mean policy_loss value_loss entropy --title "Single-Agent PPO Curves"
```

Optional:
- Set `--output-path` to control where the PNG is saved.
- Set `--x-key` if using a custom horizontal-axis metric.
