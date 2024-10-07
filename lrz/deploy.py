import argparse
import os
import json
from datetime import datetime

def config_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Script to generate and run necessary execution files"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        choices=[
            "lrz-dgx-a100-80x8",
            "lrz-v100x2",
            "lrz-dgx-1-v100x8",
            "lrz-dgx-1-p100x8",
            "lrz-hpe-p100x4",
        ],
        help="GPU partition to use",
        required=True,
    )
    parser.add_argument("--num_gpus", type=int, help="Number of GPUs to use", default=1)
    parser.add_argument(
        "--max_time", type=int, help="Maximum time for execution in minutes", default=60
    )

    return parser.parse_args()


def get_exec_str(args) -> str:
    date_time = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.now())
    return f"{args['dataset']}-{args['model_hf_key']}/{date_time}"

if __name__ == "__main__":
    parser = config_parser()
    exec_config_path = f"{parser.exec}.json"

    # read run config for folder name
    with open(exec_config_path, "r") as j:
        exec_config = json.load(j)

    # get full run path from the config json file
    exec_path = get_exec_str(exec_config)
    aug_exec_path = os.path.join("lrz", "runs", exec_path)

    # mkdir aug exec path
    if not os.path.exists(aug_exec_path):
        os.makedirs(aug_exec_path)

    config_path = os.path.join(aug_exec_path, "config.json")
    with open(config_path, "w") as fp:
        json.dump(obj=exec_config, fp=fp)

    # create dump files
    dump_out_path = os.path.join(aug_exec_path, "dump.out")
    dump_err_path = os.path.join(aug_exec_path, "dump.err")
    os.system(f"touch {dump_err_path}")
    os.system(f"touch {dump_out_path}")

    og_path_container = "/dss/dssfs04/lwp-dss-0002/t12g1/t12g1-dss-0000/"

    # create sbatch file
    sbatch_path = os.path.join(aug_exec_path, "run.sbatch")
    with open(sbatch_path, "w") as sbatch_file:
        sbatch_file.write("#!/bin/bash\n")
        sbatch_file.write("#SBATCH -N 1\n")
        sbatch_file.write(f"#SBATCH -p {parser.gpu}\n")
        sbatch_file.write(f"#SBATCH --gres=gpu:{parser.num_gpus}\n")
        sbatch_file.write("#SBATCH --ntasks=1\n")
        sbatch_file.write(f"#SBATCH -o {dump_out_path}\n")
        sbatch_file.write(f"#SBATCH -e {dump_err_path}\n")
        sbatch_file.write(f"#SBATCH --time={parser.max_time}\n\n")


        srun_command = f"srun --container-image ~/demo.sqsh --container-mounts={og_path_container}:/mnt/container torchrun --nproc_per_node={parser.num_gpus} --standalone ~/ToTpred/main.py --config ~/ToTpred/{config_path}"

        sbatch_file.write(f"{srun_command}\n")

    # submit sbatch job
    os.system(f"sbatch {sbatch_path}")
