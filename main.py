from transformers import AutoTokenizer, AutoModelForCausalLM #, BitsAndBytesConfig

import torch
from utils.wandb import wandb_init_run, wandb_push_json, wandb_push_table
import os
from config import config
from tqdm import tqdm
import sys, logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, BadRequestError
from models.regression import logistic_regression
from models.feedforward import feedforward_network
from datetime import datetime
from utils.inference import generate_prompt, generate_cot_prompt, generate_answer
from utils.ds import get_ds
CoT = False
print ("Loading .env was: ", load_dotenv())




def get_exec_str(datestamp) -> str:

    ds = llm_config['dataset'].replace("/", "-")

    exec_dir = os.path.join(config.working_dir, "runs", f"{ds}/{datestamp}")
    if not os.path.exists(exec_dir):
        os.makedirs(exec_dir)

    return exec_dir


def run_inference(ds_name):

    dataset = get_ds(ds_name)

    if llm_config["samples"] != "all":
        dataset = dataset.select([i for i in range(llm_config["samples"])])

    if llm_config["togetherai"]:
        print ("Running Inference using Together AI")
    else:
        print ("Running Inference using GPU")


    dummy = list()
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        question = sample['question']
        answer = sample['answer']

        gen_answer = generate_answer(question, togetherai=llm_config["togetherai"], tokenizer=tokenizer,
                                     CoT=CoT, model=None)

        if llm_config["verbose"]:
            # Display the question and model's chain-of-thought response
            print(f"Question: {question}")
            print(f"Model's Answer with Chain of Thought:\n{gen_answer}")
            print(f"Reference Answer:\n{answer}")

        dummy.append([question, answer, gen_answer])
    df = pd.DataFrame(dummy, columns = ['Question', 'Reference', 'Prediction'])

    # Try evaluation using GPT4
    #llm_decisions = get_gpt4_score(df["Question"].tolist(), df["Reference"].tolist(), df["Prediction"].tolist())
    #llm_decisions = [bool(decisions) for decisions in llm_decisions]
    df["llm_decisions"] =  [False] * len(df)#llm_decisions

    fname = f"{ds_name.replace('/', '-')}.xlsx"
    fname = os.path.join(get_exec_str(date_time), fname)

    df.to_excel(fname,index=False)
    print(f"Inference Results are saved to {fname}")
    print(df)
    return df

def check_class_imbalance(df: pd.DataFrame):

    true_label = len(df[df["anum_decisions"] == 1])
    false_label = len(df[df["anum_decisions"] == 0])
    print (f" True Labels: {true_label}, False Labels: {false_label}")
    print (f"LLM can generate correct answer for {(true_label / (true_label + false_label))*100}% of the samples")
    return true_label, false_label


def read_from_file(fname:str):

    path = os.path.join(config.working_dir, fname)
    df = pd.read_excel(path, )

    print (df.columns)
    if "index_original" in df.columns:
        df = df.drop("index_original", axis=1)

    df.columns = ['Question', 'Reference', 'Prediction', 'llm_decisions', 'anum_decisions']

    n_true_label, n_false_label = check_class_imbalance(df)

    if llm_config["fix_class_imbalance"]:
        df_false = df[df["anum_decisions"] == 0]
        df_true = df[df["anum_decisions"] == 1].head(n_false_label)
        df = pd.concat([df_true, df_false], ignore_index=True)
        print (f"Using only {len(df)} samples to fix class imbalance in the dataset.")
        df = df.sample(frac=1)
        df.columns = ['Question', 'Reference', 'Prediction', 'llm_decisions', 'anum_decisions']
        new_fname = os.path.basename(fname).split(".xlsx")[0] + f"_balanced_{len(df)}.xlsx"
        new_fname = os.path.join(get_exec_str(date_time), new_fname)
        df.to_excel(new_fname)
    #else:
    #    # just take samples that have human labels
    #    df_false = df[df["anum_decisions"] == 0]
    #    df_true = df[df["anum_decisions"] == 1]
    #    df = pd.concat([df_true, df_false], ignore_index=True)
    #df = df.sample(frac=1) # shuffle the rows
    return df

def get_last_token_idx(inputs_ids: list) -> list:
    last_token_idx = list()
    for input_id in inputs_ids:
        if tokenizer.pad_token_id in input_id:
            idx = input_id.index(tokenizer.pad_token_id) - 1
        else:
            idx = -1
        last_token_idx.append(idx)

    return last_token_idx

def contruct_regression_features():
    input_sentence = list(df['Question'])
    if CoT:
        input_prompts = generate_cot_prompt(input_sentence)
    else:
        input_prompts = generate_prompt(input_sentence)

    inputs = tokenizer(input_prompts, padding=True, truncation=True, return_tensors="pt")
    last_token_indices = get_last_token_idx(inputs['input_ids'].tolist())

    # Run forward pass with a batch size of 2
    # Ensure inputs are divided as per batch size
    print ("Generating Features of regression model")
    input_ids_batches = inputs['input_ids'].split(llm_config["batch_size"])
    attention_mask_batches = inputs['attention_mask'].split(llm_config["batch_size"])


    # Process each batch
    feature = None
    for input_ids, attention_mask, last_token_idx in tqdm(zip(input_ids_batches, attention_mask_batches, last_token_indices), total = len(input_ids_batches)):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        hidden_states = outputs.hidden_states
        last_layer_hidden_state = hidden_states[-1]
        last_layer_hidden_state = last_layer_hidden_state[:,last_token_idx,:]
        if feature is None:
            feature = last_layer_hidden_state
            #print (feature.size())
        else:
            #print (feature.size(), last_layer_hidden_state.size())
            feature = torch.concat((feature,last_layer_hidden_state),dim=0)



    print (feature.size())
    #X = feature.mean(dim=1)
    #print (X.size())
    feature = feature.float().numpy()
    y = pd.to_numeric(df['anum_decisions'])

    fname = os.path.join(get_exec_str(date_time), "regression_features.txt")
    np.savetxt(fname, feature, fmt='%d')
    print(f"Saved Regression Features at {fname}")

    fname = os.path.join(get_exec_str(date_time), "regression_labels.txt")
    np.savetxt(fname, np.array(y), fmt='%d')
    print(f"Saved Regression Labels at {fname}")

    return feature, y

def read_regression_features(feature_path, label_path):
    feature = np.loadtxt(feature_path, dtype=int)
    y = np.loadtxt(label_path, dtype=int)
    return feature, y


if __name__ == '__main__':

    date_time = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.now())
    llm_config = config.llm_config
    model_name = llm_config["model_hf_key"]
    print(f"Starting Script with config: {llm_config}")
    print (llm_config)
    wandb_init_run(config=llm_config)


    if llm_config["read_from_file"]:
        df = read_from_file(fname = llm_config["filename"])
    else:
        # Generate answer
        if not llm_config["togetherai"]:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                     torch_dtype=torch.bfloat16,
                                                     #load_in_4bit=True
                                                          )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        df = run_inference(llm_config["dataset"])


    if llm_config["samples"] != "all":
        if llm_config["samples"] < len(df):
            df = df.head(n=llm_config["samples"])

    
    if llm_config["regression_features_saved"]:
        feature, y = read_regression_features(llm_config["regression_features_path"], llm_config["regression_labels_path"])
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                 torch_dtype=torch.bfloat16, output_hidden_states=True,
                                                 return_dict_in_generate = True,
                                                  # load_in_8bit=True
                                                      )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        feature , y = contruct_regression_features()

    # Train the regression model.
    if llm_config["regression_model"] == "linear regression":
        accuracy, loss = logistic_regression(feature, y, llm_config )

    else:
        accuracy, loss = feedforward_network(feature, y, get_exec_str(date_time), epochs=llm_config["epochs"])


    wandb_table = {"test_accuracy": accuracy, "#sample": len(y),
                   "hidden_layer": llm_config["hidden_layer"], "reg-model": llm_config["regression_model"],
                   "balance_ds": llm_config["class_imbalance"], "epochs": llm_config["epochs"],
                    "weights_init": "HE"   }
    wandb_push_json(wandb_table)


