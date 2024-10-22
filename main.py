from transformers import AutoTokenizer, AutoModelForCausalLM #, BitsAndBytesConfig
from datasets import load_dataset
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
from together import Together
from datetime import datetime

print ("Loading .env was: ", load_dotenv())

def get_device() -> str:
    # set the device
    if torch.cuda.is_available():
        print("CUDA AVAILABLE....")
        torch.cuda.empty_cache()
        return "cuda"
    else:
        return "cpu"

# Function to generate a Chain of Thought (CoT) prompt
def generate_cot_prompt(questions):
    # Define a basic chain-of-thought prompt format

    cot_prompts = [f"Question: {question} \nLet's think step by step:\n" for question in questions]
    #print (f"CoT Prompt: {cot_prompts[0}\n")
    return cot_prompts



# Function to get the model's prediction
def generate_answer(question, togetherai = True):
    cot_prompt = generate_cot_prompt(question)
    if togetherai:
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[{"role": "user", "content": question}],
            max_tokens=512, #tokens to generate
            temperature=0,
            top_p=1,
            do_sample=False,
        )
        answer = response.choices[0].message.content
        #print("OUTPUT:", answer)
    else:
        # Tokenize the input prompt
        inputs = tokenizer(cot_prompt, return_tensors="pt", padding=True,truncation=True) #.to(device)
        #cpu_offload(model, device, offload_buffers=True)
        inputs = inputs.to(device)

        # Generate output from the model (limiting max tokens for efficiency)
        outputs = model.generate(
            **inputs,
            max_length=512,  # Adjust this to limit the length of the response
            num_beams=1,  # Beam search to improve output quality
            #early_stopping=True
            temperature = 0,
            do_sample = False,
            pad_token_id = tokenizer.eos_token_id,
        )

        # Decode the model's output
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the final answer after the step-by-step reasoning
    return answer


def get_gpt4_score(questions:list, references:list, predictions:list) -> bool:
    outputs = []
    num_rate_errors = 0
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))
    prompts = [(
        f"You are a Maths Teacher. You will be given the LLM answer to a Maths or Reasoning Question along with the correct answer. Your task is to compare the Generated Answer to the Reference Answer for the given question. "
        f"You should output True if generated answer contains has the correct output in the end, and False is the Generated Answer doesn't contain the correct answer as per the Reference Answer"
        f"\n Question: {ques}"
        f"\n Reference Answer: {ref} \n\n Generated Answer: {pred} \n Output only True or False. Evaluation: ") for ques, ref, pred in zip(questions, references, predictions)]
    for prompt in tqdm(prompts):

        response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=10,
                    temperature=0.2,
                )

        output = response.choices[0].message.content
        outputs.append(output)


    return outputs

def get_exec_str(datestamp) -> str:

    ds = llm_config['dataset'].replace("/", "-")

    exec_dir = os.path.join(config.working_dir, "runs", f"{ds}/{datestamp}")
    if not os.path.exists(exec_dir):
        os.makedirs(exec_dir)

    return exec_dir

def get_device() -> str:
    # set the device
    if torch.cuda.is_available():
        print("CUDA AVAILABLE....")
        torch.cuda.empty_cache()
        return "cuda"
    else:
        return "cpu"

def run_inference(ds_name):
    # Load the GSM8k dataset from Hugging Face
    dataset = load_dataset(ds_name, "main", split='test')
    if llm_config["samples"] != "all":
        dataset = dataset.select([i for i in range(llm_config["samples"])])
    dummy = list()
    if llm_config["togetherai"]:
        print ("Running Inference using Together AI")
    else:
        print ("Running Inference using GPU")
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        question = sample['question']
        answer = sample['answer']

        gen_answer = generate_answer(question, togetherai=llm_config["togetherai"])

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
        df.columns = ['index_original','Question', 'Reference', 'Prediction', 'llm_decisions', 'anum_decisions']
        df.to_excel("runs/openai-gsm8k/2024-10-08_20-00-58/llama3_gsm8k_balanced.xlsx")
    #else:
    #    # just take samples that have labels
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
    input_prompts = generate_cot_prompt(input_sentence)

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
        last_layer_hidden_state = hidden_states[llm_config['hidden_layer']]
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
    device = get_device()
    print(f"Starting Script with config: {llm_config}")
    print (llm_config)
    wandb_init_run(config=llm_config)


    if llm_config["read_from_file"]:
        df = read_from_file(fname = llm_config["filename"])
    else:
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
                   "balance_ds": llm_config["class_imbalance"], "epochs": llm_config["epochs"]}
    wandb_push_json(wandb_table)


