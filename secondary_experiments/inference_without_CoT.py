from os import mkdir
from transformers import AutoTokenizer, AutoModelForCausalLM, BertModel  # , BitsAndBytesConfig
import torch
from utils.wandb import wandb_init_run, wandb_push_json, wandb_push_table
from together import Together
import os
from config import config
from tqdm import tqdm
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from utils.classify_math import get_gpt4_score
from utils.inference import generate_prompt, generate_cot_prompt, generate_answer
import warnings

warnings.filterwarnings('ignore')
CoT = False
with_options = True
print ("Loading .env was: ", load_dotenv())


def get_exec_str(datestamp) -> str:
    ds = llm_config['dataset'].replace("/", "-")

    exec_dir = os.path.join(config.working_dir, "runs", f"{ds}/{datestamp}", f"CoT_{CoT}")
    if not os.path.exists(exec_dir):
        os.makedirs(exec_dir)
        os.makedirs(os.path.join(exec_dir, "models"))

    return exec_dir

def run_inference(df, ds_name):


    if llm_config["togetherai"]:
        print("Running Inference using Together AI")
    else:
        print("Running Inference using GPU")

    fname = f"{ds_name.replace('/', '-')}_wo_CoT.xlsx"
    fname = os.path.join(get_exec_str(date_time), fname)
    print(f"Inference Results would be saved to {fname}")
    generation_without_CoT = list()
    for question in tqdm(df['Question']):

        gen_answer = generate_answer(question, togetherai=llm_config["togetherai"], tokenizer=tokenizer,
                                     CoT=CoT, model=model, gen_tokens=llm_config["max_new_tokens"])

        generation_without_CoT.append(gen_answer)

    df['Prediction_wo_CoT'] = generation_without_CoT
    #df["anum_decisions_wo_CoT"] = [False] * len(df)
    df = df.drop(["anum_decisions"], axis=1)
    df["llm_decisions_wo_CoT"] = [False] * len(df)

    df.to_excel(fname, index=False)
    print(f"Inference Results without evaluation are saved to {fname}")

    #df = drop_nasty_samples(df)
    #fname = f"{ds_name.replace('/', '-')}_filtered.xlsx"
    #fname = os.path.join(get_exec_str(date_time), fname)
    #df.to_excel(fname, index=False)
    #print(f"Inference Results (cleaned) are saved to {fname}")

    # Try evaluation using GPT4
    print("Running Evaluation using GPT-4o mini.")
    df["llm_decisions_wo_CoT"] = get_gpt4_score(df["Question"].tolist(), df["Reference"].tolist(), df["Prediction_wo_CoT"].tolist())
    df.to_excel(fname, index=False)
    print(f"Inference Results with GPT-4o mini evaluation are saved to {fname}")

    return df, fname


def check_class_imbalance(df: pd.DataFrame):
    true_label = len(df[df["llm_decisions"] == 1])
    false_label = len(df[df["llm_decisions"] == 0])
    print(f" True Labels: {true_label}, False Labels: {false_label}")
    print(f"LLM can generate correct answer for {(true_label / (true_label + false_label)) * 100}% of the samples")
    return true_label, false_label


def get_balanced_ds(df, samples_per_class, fname=None):
    df_false = df[df["llm_decisions"] == 0].head(samples_per_class)
    df_true = df[df["llm_decisions"] == 1].head(samples_per_class)

    df = pd.concat([df_true, df_false], ignore_index=True)
    print(f"Using only {len(df)} samples to fix class imbalance in the dataset.")
    df = df.sample(frac=1)

    if fname is not None:
        new_fname = os.path.basename(fname).split(".xlsx")[0] + f"_balanced_{len(df)}.xlsx"
    else:
        new_fname = f"balanced_{len(df)}.xlsx"
    new_fname = os.path.join(get_exec_str(date_time), new_fname)
    df.to_excel(new_fname)
    print(f"Saved balaced dataset to {new_fname}")
    return df


def read_from_file(fname: str):
    path = os.path.join(config.working_dir, fname)
    df = pd.read_excel(path)

    print(df.columns)

    return df


def drop_nasty_samples(df):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    good_index = []
    print("Remove bad samples after inferences.")
    for i, input in tqdm(enumerate(list(df['Prediction']))):

        token = tokenizer(input, return_tensors="pt")
        token = token['input_ids']
        num_tokens = token.shape[1]
        if num_tokens <= llm_config["max_new_tokens"]:
            good_index.append(i)
    len_before = len(df)
    df = df.iloc[good_index]
    len_after = len(df)
    print(f"Dropped {len_before - len_after} samples.")
    return df

def compare_with_without_cot(df):
    analysis_log = []
    true_label = len(df[df["llm_decisions"] == 1])
    false_label = len(df[df["llm_decisions"] == 0])

    print(f" True Labels: {true_label}, False Labels: {false_label}")
    analysis_log.append(f" True Labels: {true_label}, False Labels: {false_label}")
    print(f"using CoT LLM can generate correct answer for {(true_label / (true_label + false_label)) * 100}% of the samples")
    analysis_log.append(
        f"using CoT LLM can generate correct answer for {(true_label / (true_label + false_label)) * 100}% of the samples")

    true_label = len(df[df["llm_decisions_wo_CoT"] == 1])
    false_label = len(df[df["llm_decisions_wo_CoT"] == 0])
    print(f" True Labels: {true_label}, False Labels: {false_label}")
    analysis_log.append(f" True Labels: {true_label}, False Labels: {false_label}")
    print(
        f"Without CoT LLM can generate correct answer for {(true_label / (true_label + false_label)) * 100}% of the samples")
    analysis_log.append(f"Without CoT LLM can generate correct answer for {(true_label / (true_label + false_label)) * 100}% of the samples")


    agreements = [True if a == b else False for a, b in zip(df['llm_decisions'], df['llm_decisions_wo_CoT'])]
    rate  = (agreements.count(True) / len (agreements)) * 100
    print (f"{rate}% of the time performance with and without CoT is same" )
    analysis_log.append(f"{rate}% of the time performance with and without CoT is same" )

    cot_true_no_cot_true_index = df.index[(df['llm_decisions'] == 1) & (df['llm_decisions_wo_CoT'] == 1)].tolist()
    cot_false_no_cot_false_index = df.index[(df['llm_decisions'] == 0) & (df['llm_decisions_wo_CoT'] == 0)].tolist()

    print(f"Both can solve:  {len(cot_true_no_cot_true_index)}",
          f" ({len(cot_true_no_cot_true_index) * 100 / len(df)}%)")
    analysis_log.append(f"Both can solve:  {len(cot_true_no_cot_true_index)}"
          f"{len(cot_true_no_cot_true_index) * 100 / len(df)}%)")
    print(f"both cannot solve:  {len(cot_false_no_cot_false_index)}"
          f" ({len(cot_false_no_cot_false_index) * 100 / len(df)})%")
    analysis_log.append(f" Both cannot solve:  {len(cot_false_no_cot_false_index)}"
          f" ({len(cot_false_no_cot_false_index) * 100 / len(df)})%")

    cot_true_no_cot_false_index = df.index[(df['llm_decisions'] == 1) & (df['llm_decisions_wo_CoT'] == 0)].tolist()
    cot_false_no_cot_true_index = df.index[(df['llm_decisions'] == 0) & (df['llm_decisions_wo_CoT'] == 1)].tolist()
    print(f"CoT can solve, No Cot cannot solve:  {len(cot_true_no_cot_false_index)}"
          f" ({len(cot_true_no_cot_false_index) * 100 / len(df)}%)")
    analysis_log.append(f"CoT can solve, No Cot cannot solve:  {len(cot_true_no_cot_false_index)}"
          f" ({len(cot_true_no_cot_false_index) * 100 / len(df)}%)")
    print(f"CoT could't and without Cot cannot solve: {len(cot_false_no_cot_true_index)}"
          f" ({len(cot_false_no_cot_true_index) * 100 / len(df)}%)")
    analysis_log.append(f"CoT could't and without Cot cannot solve: {len(cot_false_no_cot_true_index)}"
          f" ({len(cot_false_no_cot_true_index) * 100 / len(df)}%)")
    return analysis_log
if __name__ == '__main__':

    date_time = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.now())
    llm_config = config.llm_config
    model_name = llm_config["model_hf_key"]

    wandb_init_run(config=llm_config)


    df = read_from_file(fname=llm_config["filename"])
    new_file_name = llm_config["filename"]

    if llm_config["samples"] != "all":
        if llm_config["samples"] < len(df):
            df = df.head(n=llm_config["samples"])
    #n_true_label, n_false_label = check_class_imbalance(df)
    #samples_per_class = min(n_false_label, n_true_label)
    #df = get_balanced_ds(df, samples_per_class=samples_per_class, fname=new_file_name)



    # Generate answer
    if llm_config["togetherai"] == False:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                         torch_dtype=torch.bfloat16,
                                                         # load_in_4bit=True
                                                         )
    else:
        model = None
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    df, fname = run_inference(df, llm_config["dataset"])
    # Compare with output with and without CoT
    analysis = compare_with_without_cot(df)
    fname = f"{llm_config['dataset'].replace('/', '-')}_CoT_analysis_test.txt"
    fname = os.path.join(get_exec_str(date_time), fname)

    with open(fname, 'w') as f:
        for line in analysis:
            f.write(f"{line}\n")
    print(f"Cot analysis written at {fname}")
