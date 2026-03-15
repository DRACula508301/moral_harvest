python -m moral_harvest.cli.train \
--mode multi-agent-selfish \
--backend cleanrl \
--num-agents 2 \
--num-envs 2 \
--substrate commons_harvest__open \
--stop-iters 100 \
--checkpoint-every 50 \
--run-name 2-agent-selfish-100-iters