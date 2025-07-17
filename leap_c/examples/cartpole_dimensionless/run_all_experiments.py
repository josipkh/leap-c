import datetime
from argparse import ArgumentParser
from pathlib import Path
from leap_c.run import main
from leap_c.torch.rl.sac_fop import SacFopBaseConfig
import os
import shutil

keep_output = False  # if False, the output is saved in /tmp/

task_names = [
    "cartpole_swingup_dimensionless_default",
    "cartpole_swingup_dimensionless_small",
    "cartpole_swingup_dimensionless_large",
    "cartpole_swingup_dimensionless_small",
    "cartpole_swingup_dimensionless_large",
]

run_names = [
    "default",
    "small",
    "large",
    "transfer_small",
    "transfer_large",
]

parser = ArgumentParser()
task_name = "cartpole_swingup_dimensionless"
parser.add_argument("--task", type=str, default=task_name)
parser.add_argument("--trainer", type=str, default="sac_fop")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

output_root = "output" if keep_output else "/tmp"
time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
main_output_path = Path(f"{output_root}/{args.task}/{args.trainer}_{args.seed}_{time_str}")

def get_run_config(run_name):
    cfg = SacFopBaseConfig()
    cfg.seed = 0
    cfg.val.interval = 10_000
    cfg.train.steps = 50_000
    cfg.val.num_render_rollouts = 1
    cfg.log.wandb_logger = True
    cfg.log.csv_logger = False
    cfg.sac.entropy_reward_bonus = False  # type: ignore
    cfg.sac.update_freq = 4
    cfg.sac.batch_size = 64
    cfg.sac.lr_pi = 1e-4
    cfg.sac.lr_q = 1e-4
    cfg.sac.lr_alpha = 1e-3
    cfg.sac.init_alpha = 0.1
    cfg.sac.gamma = 1.0
    cfg.log.wandb_init_kwargs = {"name": run_name}
    return cfg

for k in range(5):
    task = task_names[k]
    output_path = Path(os.path.join(main_output_path, run_names[k]))
    cfg = get_run_config(run_name=run_names[k])

    if k > 2:
        # copy the existing folder to test transfer
        source = os.path.join(main_output_path, run_names[0])
        shutil.copytree(src=source, dst=output_path)

        # if present, remove the wandb output to initiate a clean run
        wandb_path = Path(os.path.join(output_path, "wandb"))
        if wandb_path.exists() and wandb_path.is_dir():
            shutil.rmtree(wandb_path)

    main("sac_fop", task, cfg, output_path, "cpu")

