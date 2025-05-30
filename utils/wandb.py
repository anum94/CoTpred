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

def wandb_push_json(table_json:json, i = None):
    col_names = list(table_json.keys())
    table = wandb.Table(columns=col_names)
    values = list(table_json.values())
    table.add_data(values[0], values[1], values[2], values[3], values[4],
                   values[5], values[6], values[7], values[8], values[9], values[10],
                   values[11], values[12])
    if i is None:
        wandb.log({"metrics_table": table}, commit=True)
    else:
        wandb.log({f"metrics_table_{i}": table}, commit=True)

def wandb_plot_line(x_values, y_values, title, x_caption, y_caption):

    data = [[x, y] for (x, y) in zip(x_values, y_values)]
    table = wandb.Table(data=data, columns=[x_caption, y_caption])
    wandb.log(
        {
            title: wandb.plot.line(
                table, x_caption, y_caption, title=title
            )
        }, commit=True
    )

def wandb_push_table(tab:json):
    col_names = list(tab.keys())
    table = wandb.Table(columns=col_names)
    for gen,hals in zip(tab["generations"], tab["hallucinations"]):
        h = ""
        for hal in hals:
            h = h.join("{}\\n".format(hal["atom"]))

        table.add_data(gen, h)
    wandb.log({"data_table": table}, commit=True)


def wandb_init_run( config = None, wandb_project_name = "cot-pred", entity = "anum-afzal-technical-university-of-munich"
                   ):
    wandb_setup()
    wandb_mode = "online"

    model = config["model_hf_key"].split("/")[-1]
    ds = config["dataset"].split("/")[-1]
    task = "cot"

    wandb_run_name = randomname.get_name() + '_' + '_'.join(
        [ ds])

    wandb.init(project=wandb_project_name, entity=entity, config=config, name=wandb_run_name,
               mode=wandb_mode)


