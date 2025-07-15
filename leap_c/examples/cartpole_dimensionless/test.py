import datetime
from argparse import ArgumentParser
from pathlib import Path
from leap_c.run import main
from leap_c.torch.rl.sac_fop import SacFopBaseConfig
from leap_c.examples.cartpole_dimensionless.config import dimensionless

keep_output = False  # if False, the output is saved in /tmp/

parser = ArgumentParser()
task_name = "cartpole_swingup" + ("_dimensionless" if dimensionless else "")
parser.add_argument("--task", type=str, default=task_name)
parser.add_argument("--trainer", type=str, default="sac_fop")
parser.add_argument("--output_path", type=Path, default=None)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

cfg = SacFopBaseConfig()
cfg.seed = 0
cfg.val.interval = 10_000 / 5_000
cfg.train.steps = 50_000 / 5_000
cfg.val.num_render_rollouts = 1
cfg.log.wandb_logger = False
cfg.log.wandb_init_kwargs = {"name": "test_default"}
cfg.log.csv_logger = False
cfg.sac.entropy_reward_bonus = False  # type: ignore
cfg.sac.update_freq = 4
cfg.sac.batch_size = 64
cfg.sac.lr_pi = 1e-4
cfg.sac.lr_q = 1e-4
cfg.sac.lr_alpha = 1e-3
cfg.sac.init_alpha = 0.1

# all settings:
# cfg.train.steps = 50_000
# cfg.train.start = 0
# cfg.val.interval = 10_000
# cfg.val.num_rollouts = 10
# cfg.val.deterministic = True
# cfg.val.ckpt_modus = 'best'
# cfg.val.num_render_rollouts = 1
# cfg.val.render_mode = 'rgb_array'
# cfg.val.render_deterministic = True
# cfg.val.report_score = 'cum'
# cfg.log.verbose = True
# cfg.log.interval = 1000
# cfg.log.window = 10000
# cfg.log.csv_logger = False
# cfg.log.tensorboard_logger = False
# cfg.log.wandb_logger = False
# cfg.log.wandb_init_kwargs = {}
# cfg.seed = 0
# cfg.sac.critic_mlp.hidden_dims = (256, 256, 256)
# cfg.sac.critic_mlp.activation = 'relu'
# cfg.sac.critic_mlp.weight_init = 'orthogonal'
# cfg.sac.actor_mlp.hidden_dims = (256, 256, 256)
# cfg.sac.actor_mlp.activation = 'relu'
# cfg.sac.actor_mlp.weight_init = 'orthogonal'
# cfg.sac.batch_size = 64
# cfg.sac.buffer_size = 1000000
# cfg.sac.gamma = 0.99
# cfg.sac.tau = 0.005
# cfg.sac.soft_update_freq = 1
# cfg.sac.lr_q = 1e-4
# cfg.sac.lr_pi = 3e-4
# cfg.sac.lr_alpha = 1e-4
# cfg.sac.init_alpha = 0.1
# cfg.sac.target_entropy = None
# cfg.sac.entropy_reward_bonus = False
# cfg.sac.num_critics = 2
# cfg.sac.report_loss_freq = 100
# cfg.sac.update_freq = 4
# cfg.noise = "param"

output_root = "output" if keep_output else "/tmp"
time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
output_path = Path(f"{output_root}/{args.task}/{args.trainer}_{args.seed}_{time_str}")

# for testing transfer learning, set the output path to an existing directory
# output_path = Path("/home/josip/leap-c/output/cartpole_swingup_dimensionless/sac_fop_0_20250704102054_transfer_2")

main(args.trainer, args.task, cfg, output_path, args.device)
