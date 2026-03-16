python -m moral_harvest.cli.train \
--mode multi-agent-selfish \
--backend cleanrl \
--num-agents 10 \
--num-envs 2 \
--substrate commons_harvest__open \
--stop-iters 300 \
--checkpoint-every 100 \
--ent-coef 0.1 \
--run-name selfish_10_agent_300_iters