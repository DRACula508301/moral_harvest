python -m moral_harvest.cli.train \
--mode multi-agent-selfish \
--backend cleanrl \
--num-agents 7 \
--substrate commons_harvest__open \
--stop-iters 2000 \
--num-envs 8 \
--train-batch-size 1000 \
--minibatch-size 1000 \
--checkpoint-every 100 \
--ent-coef 0.02

# These flags apply only to --mode multi-agent-reward-shaped.
# --shaping-begin 300000
# --rew-shaping-horizon 700000

# Note: shaping schedule flags apply to --mode multi-agent-reward-shaped only.
# --shaping-begin 1600000
# --rew-shaping-horizon 2400000