import datetime
import os
import shutil
from pathlib import Path
from leap_c.run import main
from leap_c.torch.rl.sac_fop import SacFopBaseConfig
from leap_c.examples.cartpole_dimensionless.task import CartpoleSwingupDimensionless
from leap_c.examples.cartpole_dimensionless.config import get_default_cartpole_params
from leap_c.examples.cartpole_dimensionless.utils import get_similar_cartpole_params, plot_results

# high-level experiment settings
keep_output = False  # if False, the output is saved in /tmp/
task_name = "cartpole_swingup_dimensionless"
trainer_name = "sac_fop"
device = "cpu"
output_root = "output" if keep_output else "/tmp"
time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
main_output_path = Path(f"{output_root}/{task_name}/{time_str}")

# cartpole parameters
default_params = get_default_cartpole_params()
small_params = get_similar_cartpole_params(reference_params=default_params, rod_length=0.1)
large_params = get_similar_cartpole_params(reference_params=default_params, rod_length=5.0)

# learning tasks
task_default = CartpoleSwingupDimensionless(mpc_params=default_params, env_params=default_params)
task_small = CartpoleSwingupDimensionless(mpc_params=small_params, env_params=small_params)
task_large = CartpoleSwingupDimensionless(mpc_params=large_params, env_params=large_params)

task_list = [
    task_default,
    task_small,
    task_large,
    task_small,
    task_large,
]

run_names = [
    "default",
    "small",
    "large",
    "transfer_small",
    "transfer_large",
]

# learning configuration
def get_run_config(run_name, seed):
    cfg = SacFopBaseConfig()
    cfg.seed = seed
    cfg.val.interval = 10_000
    cfg.train.steps = 50_000
    cfg.val.num_render_rollouts = 1
    cfg.log.wandb_logger = True
    cfg.log.csv_logger = True
    cfg.sac.entropy_reward_bonus = False  # type: ignore
    cfg.sac.update_freq = 4
    cfg.sac.batch_size = 64
    cfg.sac.lr_pi = 1e-4
    cfg.sac.lr_q = 1e-4
    cfg.sac.lr_alpha = 1e-3
    cfg.sac.init_alpha = 0.1
    cfg.sac.gamma = 1.0
    cfg.log.wandb_init_kwargs = {"name": run_name + "-" + str(seed)}
    return cfg

# main loop
for seed in range(5):
    for k in range(5):
        task = task_list[k]
        run_name = run_names[k]
        output_path = Path(os.path.join(main_output_path, run_name, str(seed)))
        cfg = get_run_config(run_name, seed)

        if k > 2:
            # copy the existing folder to test transfer
            source = os.path.join(main_output_path, run_names[0], str(seed))
            shutil.copytree(src=source, dst=output_path)

            # if present, remove the wandb output to initiate a clean run
            wandb_path = Path(os.path.join(output_path, "wandb"))
            if wandb_path.exists() and wandb_path.is_dir():
                shutil.rmtree(wandb_path)

        main(trainer_name, task_name, cfg, output_path, device, task)

plot_results(main_folder=main_output_path, cfg=cfg, plot_std=False)
