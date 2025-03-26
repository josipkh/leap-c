from argparse import ArgumentParser
from pathlib import Path
import csv

from leap_c.run import main
from leap_c.rl.sac_fop import SacFopBaseConfig


parser = ArgumentParser()
parser.add_argument("--output_path", type=Path, default=None)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

output_path = Path(f"output/vehicle_steering/sac_fop_{args.seed}")


def get_last_val():
    path = output_path / "val_log.csv"
    vals = []
    with path.open() as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            try:
                final_val = float(row[0].split(',')[1])
                vals.append(final_val)
            except:
                pass
    return vals[-1]


def run_learning(lr_pi, lr_q, lr_alpha):
    cfg = SacFopBaseConfig()
    cfg.val.interval = 500
    cfg.train.steps = 1_000
    cfg.val.num_render_rollouts = 1
    cfg.log.wandb_logger = False
    cfg.log.tensorboard_logger = False
    cfg.sac.entropy_reward_bonus = False  # type: ignore
    cfg.sac.update_freq = 4
    cfg.sac.batch_size = 64
    cfg.sac.lr_pi = lr_pi  # 1e-4
    cfg.sac.lr_q = lr_q  # 1e-4
    cfg.sac.lr_alpha = lr_alpha  # 1e-3
    cfg.sac.init_alpha = 0.10

    main("sac_fop", "vehicle_steering_stabilization", cfg, output_path, args.device)

    return get_last_val()


if __name__ == "__main__":
    import optuna

    def objective(trial):
        lr_pi = trial.suggest_float('lr_pi', 1e-4, 1e-3)
        lr_q = trial.suggest_float('lr_q', 1e-4, 1e-3)
        lr_alpha = trial.suggest_float('lr_alpha', 1e-4, 1e-3)
        return run_learning(lr_pi, lr_q, lr_alpha)

    study = optuna.create_study()
    study.optimize(objective, n_trials=5)

    print(study.best_params)