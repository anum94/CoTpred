import randomname as randomname
import os
import wandb
import json

def wandb_setup():

    wandb_token_key: str = "WANDB_TOKEN"

    # wandb setup
    wandb_tok = os.environ.get(wandb_token_key)
    assert wandb_tok and wandb_tok != "<wb_token>", "Wandb token is not defined"
    wandb.login( key=wandb_tok)

def wandb_push_json(table_json:json):
    col_names = list(table_json.keys())
    table = wandb.Table(columns=col_names)
    values = list(table_json.values())
    table.add_data(values[0], values[1], values[2])
    wandb.log({"metrics_table": table}, commit=True)

def wandb_push_table(tab:json):
    col_names = list(tab.keys())
    table = wandb.Table(columns=col_names)
    for gen,hals in zip(tab["generations"], tab["hallucinations"]):
        h = ""
        for hal in hals:
            h = h.join("{}\\n".format(hal["atom"]))

        table.add_data(gen, h)
    wandb.log({"data_table": table}, commit=True)


def wandb_init_run( config = None, wandb_project_name = "cot-pred", entity = "anum-afzal-technical-university-of-munich-org"
                   ):
    wandb_setup()
    wandb_mode = "online"

    model = config["model_hf_key"].split("/")[-1]
    ds = config["dataset"].split("/")[-1]
    task = "cot"

    wandb_run_name = randomname.get_name() + '_' + '_'.join(
        [model, ds, task])

    wandb.init(project=wandb_project_name, entity=entity, config=vars(config), name=wandb_run_name,
               mode=wandb_mode, group=config["samples"])


