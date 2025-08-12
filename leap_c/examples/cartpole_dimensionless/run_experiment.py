import datetime
from pathlib import Path
from leap_c.run import main
from leap_c.torch.rl.sac_fop import SacFopBaseConfig
from leap_c.examples.cartpole_dimensionless.task import CartpoleSwingupDimensionless
from leap_c.examples.cartpole_dimensionless.config import get_default_cartpole_params
from leap_c.examples.cartpole_dimensionless.utils import get_similar_cartpole_params

keep_output = False  # if False, the output is saved in /tmp/
dimensionless = True

trainer_name = "sac_fop"
task_name = "cartpole_swingup" + ("_dimensionless" if dimensionless else "")
device = "cpu"
task = CartpoleSwingupDimensionless(
    mpc_params=get_default_cartpole_params(),
    env_params=get_similar_cartpole_params(
        reference_params=get_default_cartpole_params(),
        rod_length=5.0,
    ),
)
seed = 0

output_root = "output" if keep_output else "/tmp"
time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
output_path = Path(f"{output_root}/{task_name}/{trainer_name}_{seed}_{time_str}")

# for testing transfer learning, set the output path to an existing directory
# output_path = Path("/home/josip/leap-c/output/cartpole_swingup_dimensionless/sac_fop_0_20250704102054_transfer_2")

cfg = SacFopBaseConfig()
cfg.seed = seed
cfg.val.interval = 10_000
cfg.train.steps = 50_000
cfg.val.num_render_rollouts = 1
cfg.log.wandb_logger = True
cfg.log.wandb_init_kwargs = {"name": "test"}
cfg.log.csv_logger = False
cfg.sac.entropy_reward_bonus = False  # type: ignore
cfg.sac.update_freq = 4
cfg.sac.batch_size = 64
cfg.sac.lr_pi = 1e-4
cfg.sac.lr_q = 1e-4
cfg.sac.lr_alpha = 1e-3
cfg.sac.init_alpha = 0.1
cfg.sac.gamma = 1.0

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

main(trainer_name, task_name, cfg, output_path, device, task)
