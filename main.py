from os import mkdir
from pyexpat import features

from pyarrow.dataset import dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BertModel #, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets
import torch
from utils.wandb import wandb_init_run, wandb_push_json, wandb_push_table
from together import Together
import os
from config import config
from tqdm import tqdm
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from models.regression import logistic_regression
from models.feedforward import feedforward_network
from datetime import datetime
from os import listdir
from os.path import isfile, join, split
from utils.classify_math import get_gpt4_score
from utils.inference import generate_prompt, generate_cot_prompt, generate_answer
import warnings
warnings.filterwarnings('ignore')
CoT = True
with_options = True
#print ("Loading .env was: ", load_dotenv())


def get_exec_str(datestamp) -> str:

    ds = llm_config['dataset'].replace("/", "-")

    exec_dir = os.path.join(config.working_dir, "runs", f"{ds}/{datestamp}", f"CoT_{CoT}")
    if not os.path.exists(exec_dir):
        os.makedirs(exec_dir)
        os.makedirs(os.path.join(exec_dir, "models"))

    return exec_dir

def get_ds(ds_name):


    if ds_name == "openai/gsm8k":
        #dataset_train = load_dataset("openai/gsm8k", "main", split='train')
        #dataset_test = load_dataset("openai/gsm8k", "main", split='test')
        #dataset = concatenate_datasets([dataset_train, dataset_test])
        dataset = load_dataset("openai/gsm8k", "main", split='test')
        return dataset
    elif ds_name == "aops_forum":
        dataset = load_dataset("json", data_files="aops_forum.json", split="train")
        dataset= dataset.rename_column("ground_truth", "answer")

        return dataset

    elif ds_name =="olympiad":
        def fn_numina(sample, _):
            return {"question": sample["problem"], "answer": sample["solution"]}
        dataset = load_dataset("AI-MO/NuminaMath-CoT",  split='train')
        dataset = dataset.filter(lambda example: example['source'].startswith('olympiads'))
        return dataset.map(fn_numina, dataset, batched=True, remove_columns=["messages", "source", "problem", "solution"])

    elif ds_name == "lighteval/MATH":
        def fn_math(sample, _):
            return {"question": sample["problem"], "answer": sample["solution"]}
        dataset_train = load_dataset("lighteval/MATH", split="train", trust_remote_code=True)
        dataset_test = load_dataset("lighteval/MATH", split="test", trust_remote_code=True)
        dataset = concatenate_datasets([dataset_train, dataset_test])
        return dataset.map(fn_math, dataset, batched=True, remove_columns=["type", "level", "problem", "solution"])

    elif ds_name == "aslawliet/cn-k12":
        def fn_cnk12(sample, _):
            return {"question": sample["problem"], "answer": sample["solution"]}
        dataset= load_dataset("aslawliet/cn-k12", split="train", trust_remote_code=True)
        return dataset.map(fn_cnk12, dataset, batched=True, remove_columns=[ "problem", "solution"])

    elif ds_name == "deepmind/aqua_rat":
        dataset = load_dataset("deepmind/aqua_rat", "tokenized", split='train')

        def fn_aquarat(sample, _):
            prompt = "\n One of the following is the correct answer. \n"
            options = [' \n'.join(i) for i in sample['options']]

            if with_options:
                question = [q + prompt + o for q, o in zip(sample['question'], options)]

                return {"question": question, "answer": sample["rationale"], "correct_option": sample["correct"]}
            else:
                answer = [ prompt + o + a for a, o in zip(sample['rationale'], options)]

                return {"question": sample["question"], "answer": answer, "correct_option": sample["correct"]}


        return dataset.map(fn_aquarat, dataset, batched=True, remove_columns=["rationale", "correct"])


def run_inference(ds_name):

    dataset = get_ds(ds_name)

    if llm_config["samples"] != "all":
        dataset = dataset.select([i for i in range(llm_config["samples"])])
    #df = pd.read_excel("runs/processed_ds/deepmind-aqua_rat/test_set/balanced_1044_6k_200_labelled.xlsx")
    #index = df['Unnamed: 0'].tolist()
    #index = [i  for i in range(93000, len(df))]
    #dataset = dataset.select(index)
    #dataset = dataset.select([i for i in range(93000, len(df))])


    if llm_config["togetherai"]:
        print ("Running Inference using Together AI")
    else:
        print ("Running Inference using GPU")

    fname = f"{ds_name.replace('/', '-')}.xlsx"
    fname = os.path.join(get_exec_str(date_time), fname)
    print (f"Inference Results would be saved to {fname}")
    dummy = list()
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        question = sample['question']
        answer = sample['answer']

        gen_answer = generate_answer(question, togetherai=llm_config["togetherai"], tokenizer=tokenizer,
                                     CoT=CoT, model=model, gen_tokens = llm_config["max_new_tokens"])

        if llm_config["verbose"]:
            # Display the question and model's chain-of-thought response
            print(f"Question: {question}")
            print(f"Model's Answer with Chain of Thought:\n{gen_answer}")
            print(f"Reference Answer:\n{answer}")

        dummy.append([question, answer, gen_answer])

        if (len(dummy) % 500) == 0:
            df = pd.DataFrame(dummy, columns=['Question', 'Reference', 'Prediction'])
            df["anum_decisions"] = [False] * len(df)
            df["llm_decisions"] = [False] * len(df)

            df.to_excel(fname, index=False)


    df = pd.DataFrame(dummy, columns = ['Question', 'Reference', 'Prediction'])
    df["anum_decisions"] = [False] * len(df)
    df["llm_decisions"] = [False] * len(df)

    df.to_excel(fname, index=False)
    print(f"Inference Results without evaluation are saved to {fname}")


    df = drop_nasty_samples(df)
    fname = f"{ds_name.replace('/', '-')}_filtered.xlsx"
    fname = os.path.join(get_exec_str(date_time), fname)
    df.to_excel(fname, index=False)
    print(f"Inference Results (cleaned) are saved to {fname}")

    # Try evaluation using GPT4
    print("Running Evaluation using GPT-4o mini.")
    df["llm_decisions"] = get_gpt4_score(df["Question"].tolist(), df["Reference"].tolist(), df["Prediction"].tolist())
    df.to_excel(fname,index=False)
    print(f"Inference Results with GPT-4o mini evaluation are saved to {fname}")


    return df, fname

def check_class_imbalance(df: pd.DataFrame):

    true_label = len(df[df["llm_decisions"] == 1])
    false_label = len(df[df["llm_decisions"] == 0])
    print (f" True Labels: {true_label}, False Labels: {false_label}")
    print (f"LLM can generate correct answer for {(true_label / (true_label + false_label))*100}% of the samples")
    return true_label, false_label

def get_balanced_ds(df, samples_per_class, fname = None):


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
def read_from_file(fname:str):

    path = os.path.join(config.working_dir, fname)
    df = pd.read_excel(path)

    print (df.columns)

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

def contruct_regression_features(df, date_time, compute_all = False):

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
    features = None
    for input_ids, attention_mask, last_token_idx in tqdm(zip(input_ids_batches, attention_mask_batches, last_token_indices), total = len(input_ids_batches)):
        with torch.no_grad():
            input_ids = input_ids.to('cuda')
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        hidden_states = outputs.hidden_states

        if compute_all: # save for all hidden layers
            hidden_states = torch.stack(list(hidden_states), dim=0)
            last_token_reps = []
            for hidden_state in hidden_states:
                last_token_rep = hidden_state[:,last_token_idx,:]
                last_token_reps.append(last_token_rep)

            if features is None:
                last_token_reps = torch.stack(last_token_reps, dim=0).unsqueeze(0)
            else:
                last_token_reps = torch.stack(last_token_reps, dim=0).unsqueeze(0)

        else: #just compute for one layer
            last_layer_hidden_state = hidden_states[llm_config['hidden_layer']]
            last_token_reps = last_layer_hidden_state[:,last_token_idx,:]


        if features is None:
            features = last_token_reps
                #print (feature.size())
        else:
            #print (feature.size(), last_layer_hidden_state.size())
            features = torch.concat((features,last_token_reps),dim=0)

    features_paths = []
    if compute_all:

        features = features.squeeze()
        a,b,c = features.size()
        for i in range(b):
            feature = features[:,i,:]
            feature = feature.float().numpy()
            fname = os.path.join(get_exec_str(date_time), f"regression_features_layer_{i}.txt")
            np.savetxt(fname, feature, fmt='%.8f')
            print(f"Saved Regression Features at {fname}")
            features_paths.append(fname)

    else:
        fname = os.path.join(get_exec_str(date_time), "regression_features.txt")
        features = features.float().numpy()
        np.savetxt(fname, features, fmt='%.8f')
        print(f"Saved Regression Features at {fname}")
        features_paths.append(fname)

    y = pd.to_numeric(df['llm_decisions'])
    fname = os.path.join(get_exec_str(date_time), "regression_labels.txt")
    np.savetxt(fname, np.array(y), fmt='%d')
    print(f"Saved Regression Labels at {fname}")

    return features_paths, y
def drop_nasty_samples(df):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    good_index = []
    print ("Remove bad samples after inferences.")
    for i, input in tqdm(enumerate(list(df['Prediction']))):

        token = tokenizer(input, return_tensors="pt")
        token = token['input_ids']
        num_tokens = token.shape[1]
        if num_tokens <= llm_config["max_new_tokens"]:
            good_index.append(i)
    len_before = len(df)
    df = df.iloc[good_index]
    len_after = len(df)
    print (f"Dropped {len_before-len_after} samples.")
    return df

def get_all_feature_paths(path):

    files = [f for f in listdir(path) if isfile(join(path, f))]
    files = [os.path.join(path,f) for f in files if "regression_features_layer_" in f]
    return files
def read_regression_features(feature_path, label_path):

    feature = np.loadtxt(feature_path, dtype=float)

    y = np.loadtxt(label_path, dtype=int)
    return feature, y

def get_baseline_features(df):
    features = []
    #df = df.head(20)
    model_name = llm_config["baseline_model"]
    print (f"Generating Embedding using Together AI using {model_name}")

    if llm_config["togetherai"] == False:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)


    input_sentence = list(df['Question'])
    if CoT:
        input_prompts = generate_cot_prompt(input_sentence)
    else:
        input_prompts = generate_prompt(input_sentence)
    for question in tqdm(input_prompts):
        if llm_config["togetherai"]:
            client = Together(
                api_key=os.environ.get("TOGETHER_API_KEY"))
            response = client.embeddings.create(
                model=model_name,
                input=question
            )
            embedding = response.data[0].embedding
        else:
            # Encode query and documents
            inputs = tokenizer(question, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            embedding = last_hidden_states[:,0,:]
            embedding = embedding.squeeze()

        features.append(embedding)

    os.makedirs(os.path.join(get_exec_str(date_time),model_name))

    fname = os.path.join(get_exec_str(date_time),model_name, "regression_labels.txt")
    y = np.array(df['llm_decisions'])
    np.savetxt(fname, y, fmt='%d')
    print(f"Saved Regression Labels at {fname}")

    fname = os.path.join(get_exec_str(date_time),model_name, "regression_features.txt")
    features = np.array(features)
    np.savetxt(fname, features, fmt='%.8f')
    print(f"Saved Regression Features at {fname}")


    return [fname], y
if __name__ == '__main__':

    date_time = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.now())
    llm_config = config.llm_config
    model_name = llm_config["model_hf_key"]
    print(f"Starting Script with config: {llm_config}")
    print (llm_config)
    wandb_init_run(config=llm_config)

    if llm_config["read_from_file"]:
        df = read_from_file(fname = llm_config["filename"])
        new_file_name = llm_config["filename"]

    else:
        # Generate answer
        if llm_config["togetherai"] == False:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                     torch_dtype=torch.bfloat16,
                                                     #load_in_4bit=True
                                                          )
        else:
            model = None
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        df, fname = run_inference(llm_config["dataset"])
        new_file_name = fname

    #df = drop_nasty_samples(df)
    if llm_config["fix_class_imbalance"]:
        n_true_label, n_false_label = check_class_imbalance(df)
        samples_per_class = min(n_false_label, n_true_label)
        df = get_balanced_ds(df, samples_per_class=samples_per_class, fname=new_file_name)

    if llm_config["samples"] != "all":
        if llm_config["samples"] < len(df):
            df = df.head(n=llm_config["samples"])

    if llm_config["baseline"]:
        # get baseline features
        if llm_config["baseline_features_saved"]:
            regression_feature_paths = [llm_config["baseline_regression_features_path"]]

        else:
            regression_feature_paths, y = get_baseline_features(df)

        wandb_table = { "#sample": len(df),
                       "hidden_layer": "N/A",
                        "reg-model": llm_config["regression_model"],
                        "batch_size": llm_config["batch_size"],
                        "epochs": llm_config["epochs"],
                       "weights_init": "HE",
                        "CoT": llm_config["baseline_model"]}
        test_features_path = llm_config["baseline_test_features_path"]

    else:

        if llm_config["regression_features_saved"]: # read previously generated features
            if llm_config["all_hidden_layers"]:
                regression_feature_paths = get_all_feature_paths(llm_config["regression_features_path"])

            else:
                regression_feature_paths = [llm_config["regression_features_path"]]
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                     torch_dtype=torch.bfloat16, output_hidden_states=True,
                                                     return_dict_in_generate = True,
                                                      # load_in_8bit=True
                                                          )

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            features , y = contruct_regression_features(df, date_time=date_time, compute_all=llm_config["all_hidden_layers"])
            if llm_config["all_hidden_layers"] == False:
                features = [features]
        wandb_table = {
                           "hidden_layer": llm_config["hidden_layer"],
                       "reg-model": llm_config["regression_model"],
                           "batch_size": llm_config["batch_size"],
                       "epochs": llm_config["epochs"],
                           "weights_init": "HE",
                       "CoT": CoT}
        test_features_path = llm_config["test_features_path"]

    # Train the regression model.

    test_label_path = llm_config["test_filename"]
    #features = [features]
    scores = None
    best_score = None
    batch_size = [32, 64, 128] # [8,16,32,64]
    weights_init = [ 'HE_uniform' ,None] #'HE_normal',
    learning_rate = [0.001,]# 0.01, 0.0001]
    thresholds = [0.5, 0.6,]# 0.6,]# 0.75]
    human_labelled = [True]#, False]
    optimizers = ['adam', 'sgd']

    for human in tqdm(human_labelled):
        for th in tqdm(thresholds):
            for lr in tqdm(learning_rate):
                for bs in tqdm(batch_size):
                    for w_init in tqdm(weights_init):
                        for optimizer in tqdm(optimizers):
                            for i, hidden_layer_path in tqdm(enumerate(regression_feature_paths), total = len(regression_feature_paths)):
                                wandb_table["hidden_layer"] = os.path.basename(hidden_layer_path)
                                wandb_table["batch_size"] = bs
                                wandb_table["weights_init"] = w_init
                                wandb_table["learning_rate"] = lr
                                wandb_table["optimizer"] = optimizer
                                wandb_table["human_labelled"] = human
                                wandb_table["threshold"] = th


                                feature, y = read_regression_features(hidden_layer_path, llm_config["regression_labels_path"])
                                if llm_config["regression_model"] == "linear regression":
                                    accuracy, loss = logistic_regression(feature, y, llm_config )

                                else:
                                    accuracy, loss = feedforward_network(feature, y, get_exec_str(date_time), epochs=llm_config["epochs"],
                                                                             i = wandb_table["hidden_layer"], batch_size=bs, weights_init=w_init, lr= lr,
                                                                             external_test_set=True, confidence_th=th, optimizer=optimizer,
                                                                         test_features_path=test_features_path,test_label_path=test_label_path ,
                                                                         human_annotation=human, baseline = llm_config["baseline"],
                                                                         PoT = llm_config["PoT"])

                                wandb_table["avg_accuracy"] = accuracy["accuracy"]
                                accuracy.pop('accuracy')
                                wandb_table["t_accuracy"]= accuracy
                                wandb_table["#sample"]= len(y)
                                if scores is None:
                                    scores = pd.DataFrame.from_dict([wandb_table])
                                else:
                                    scores = pd.concat([scores,pd.DataFrame.from_records([wandb_table]) ], axis = 0)
                                wandb_push_json(wandb_table, i=i)
    scores.to_excel(f"hp_optimization_scores_{llm_config['dataset'].replace('/', '-')}_baseline_{llm_config['baseline']}_PoT_{llm_config['PoT']}.xlsx")


