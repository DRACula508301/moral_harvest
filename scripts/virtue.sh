.venv-linux/bin/python -m moral_harvest.cli.train \
    --mode multi-agent-reward-shaped \
    --reward-type virtue \
    # --shaping-begin 1600000 \
    # --rew-shaping-horizon 2400000 \
    --stop-iters 300 \
    --checkpoint-every 100