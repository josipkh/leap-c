"""Main script to run experiments."""
import datetime
from argparse import ArgumentParser
from pathlib import Path

from leap_c.run import main
from leap_c.torch.rl.sac_fop import SacFopBaseConfig


parser = ArgumentParser()
parser.add_argument("--output_path", type=Path, default=None)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


cfg = SacFopBaseConfig()
cfg.val.interval = 10_000
cfg.train.steps = 50_000
cfg.val.num_render_rollouts = 1
cfg.log.wandb_logger = False
cfg.log.tensorboard_logger = False
cfg.sac.entropy_reward_bonus = False  # type: ignore
cfg.sac.update_freq = 4
cfg.sac.batch_size = 64
cfg.sac.lr_pi = 1e-4
cfg.sac.lr_q = 1e-4
cfg.sac.lr_alpha = 1e-3
cfg.sac.init_alpha = 0.1


output_path = Path(f"/tmp/cartpole/sac_fop_{args.seed}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")

main("sac_fop", "cartpole_swingup", cfg, output_path, args.device)
