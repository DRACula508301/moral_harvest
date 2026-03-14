# moral_harvest

PPO in Melting Pot Harvest with reward shaping.

## Single-agent PPO (Step 1)

The current runnable path is a **single PPO focal agent** in `commons_harvest__open`.
The wrapper keeps only local observations (`RGB` by default) and assigns no-op actions to all non-focal agents.

Both backends are available:
- `rllib` (default): RLlib PPO implementation.
- `cleanrl`: local CleanRL-style PPO implementation.

### Run from WSL using `.venv-linux`

From Windows PowerShell:

```powershell
wsl
```

```bash
cd /mnt/c/Users/dchen/Documents/Projects/moral_harvest && .venv-linux/bin/python -m moral_harvest.cli.train --mode single-agent --backend rllib --substrate commons_harvest__open --focal-agent player_0 --stop-iters 300 --checkpoint-every 100 --checkpoint-root checkpoints/single_agent/rllib
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
