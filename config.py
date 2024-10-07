import argparse
import time
import os
from json import load
from typing import Sequence
import wandb
from dotenv import load_dotenv
from huggingface_hub.hf_api import ModelInfo
import torch


class Config:

    hf_token_key: str = "HF_API_TOKEN"
    hf_user_key: str = "HF_USER"
    hf_hub_cache_key: str = "TRANSFORMERS_CACHE"
    hf_ds_cache_key: str = "HF_DATASETS_CACHE"
    container_prefix: str = "/mnt/container/"

    hf_hub_cache_path: str = os.path.join(container_prefix, ".cache/huggingface/hub")
    hf_ds_cache_path: str = os.path.join(
        container_prefix, ".cache/huggingface/datasets"
    )

    wandb_token_key: str = "WANDB_TOKEN"
    wandb_log_key: str = "WANDB_LOG_MODEL"
    wandb_watch_key: str = "WANDB_WATCH"

    exec_args: dict = {}
    exec_kwargs: dict = {}
    exec_string: str = ""
    exec_timestamp: str = ""


    hf_token: str = ""
    hf_user: str = ""

    def __init__(self) -> None:


        self.parse_args(self.configure_parser())
        self.configure_env()

        if self.exec_args: #does execute this if running for llm script
            self.exec_string = f"{self.llm_config['model']}-{self.llm_config['dataset']}"
        self.exec_timestamp = time.strftime("%d%m%Y%H%M%S", time.localtime())

        self.working_dir = os.path.dirname(os.path.abspath(__file__))


    def configure_parser(self) -> dict:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, help="configuration file path")
        parser.add_argument("--model", type=str, help="model name")
        parser.add_argument("--dataset", type=str, help="data name")
        return vars(parser.parse_args())

    def parse_args(self, args: dict) -> None:
        if args["config"]:
            with open(args["config"], "r") as fp:
                config = load(fp)
                self.llm_config = config["llm_config"]
                self.wandb_config = config["wandb_config"]

    def configure_env(self) -> str:
        load_dotenv()
        # hf setup
        token = os.environ.get(self.hf_token_key)
        assert token and token != "<token>", "HuggingFace API token is not defined"
        user = os.environ.get(self.hf_user_key)

        os.environ[self.hf_ds_cache_key] = self.hf_ds_cache_path
        os.environ[self.hf_hub_cache_key] = self.hf_hub_cache_path

        # wandb setup
        wandb_tok = os.environ.get(self.wandb_token_key)
        assert wandb_tok and wandb_tok != "<wb_token>", "Wandb token is not defined"
        wandb.login(anonymous="allow", key=wandb_tok)

        # os.environ["WANDB_PROJECT"]="my-awesome-project"
        os.environ[self.wandb_log_key] = "false"
        os.environ[self.wandb_watch_key] = "false"

    def ssh_key_path(self) -> str:
        return f"{self.working_dir}/.ssh/id_ed25519"

    def log_path(self) -> str:
        return f"{self.working_dir}/results/{self.exec_string}-{self.exec_timestamp}"


config = Config()
