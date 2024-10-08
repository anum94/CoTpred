from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import torch
import gc
import os
from config import config
from tqdm import tqdm
import sys, logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, BadRequestError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, classification_report

from datetime import datetime
from sklearn.metrics import mean_squared_error
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
def generate_answer(question):
    cot_prompt = generate_cot_prompt(question)

    # Tokenize the input prompt
    inputs = tokenizer(cot_prompt, return_tensors="pt") #.to(device)
    #cpu_offload(model, device, offload_buffers=True)

    # Generate output from the model (limiting max tokens for efficiency)
    outputs = model.generate(
        **inputs,
        max_length=512,  # Adjust this to limit the length of the response
        num_beams=1,  # Beam search to improve output quality
        #early_stopping=True
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
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        question = sample['question']
        answer = sample['answer']

        gen_answer = generate_answer(question)

        # Display the question and model's chain-of-thought response
        print(f"Question: {question}")
        print(f"Model's Answer with Chain of Thought:\n{gen_answer}")
        print(f"Reference Answer:\n{answer}")
        dummy.append([question, answer, gen_answer])
    df = pd.DataFrame(dummy, columns = ['Question', 'Reference', 'Prediction'])

    fname = f"{ds_name.replace('/', '-')}.xlsx"
    fname = os.path.join(get_exec_str(date_time), fname)

    df.to_excel(fname,index=False)
    logging.info (f"Inference Results are saved to {fname}")
    return df

def read_from_file(fname:str):

    df = pd.read_excel(fname, )
    print (df.columns)
    df.columns = ['Question', 'Reference', 'Prediction', 'llm_decisions', 'anum_decisions']
    return df

def contruct_regression_features():
    input_sentence = list(df['Question'])
    input_prompts = generate_cot_prompt(input_sentence)

    inputs = tokenizer(input_prompts, padding=True, truncation=True, return_tensors="pt")

    # Run forward pass with a batch size of 2
    # Ensure inputs are divided as per batch size
    input_ids_batches = inputs['input_ids'].split(llm_config["batch_size"])
    attention_mask_batches = inputs['attention_mask'].split(llm_config["batch_size"])


    # Process each batch
    for input_ids, attention_mask in zip(input_ids_batches, attention_mask_batches):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)


        #outputs = model(**inputs)
        hidden_states = outputs.hidden_states
        last_layer_hidden_state = hidden_states[-1]
        last_layer_hidden_state = last_layer_hidden_state.mean(axis=1)
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
    logging.info(f"Saved Regression Features at {fname}")

    fname = os.path.join(get_exec_str(date_time), "regression_labels.txt")
    np.savetxt(fname, np.array(y), fmt='%d')
    logging.info(f"Saved Regression Labels at {fname}")

    return feature, y

def read_regression_features(feature_path, label_path):
    feature = np.loadtxt(feature_path, dtype=int)
    y = np.loadtxt(label_path, dtype=int)
    return feature, y

def logistic_regression(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':

    date_time = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.now())
    llm_config = config.llm_config
    model_name = llm_config["model_hf_key"]
    device = get_device()
    logging.info(f"Starting Script with config: {llm_config}")
    print (llm_config)

    quantization_config = BitsAndBytesConfig(load_in_4bit=llm_config["load_in_4bit"],load_in_8_bit=llm_config["load_in_8bit"])

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                 torch_dtype=torch.bfloat16, output_hidden_states=True,
                                                 quantization_config=quantization_config)  # .to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id


    if llm_config["read_from_file"]:
        df = read_from_file(fname = llm_config["filename"])
    else:
        df = run_inference(llm_config["dataset"])


    if llm_config["samples"] != "all":
        df = df.head(n=llm_config["samples"])

    if llm_config["regression_features_saved"]:
        pass # read from file
    else:
        feature , y = contruct_regression_features()

   #Some memory cleanup
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Train the regression model.
    logistic_regression(feature, y )

    '''
    tokenizer = None
    model = None
    df = pd.read_csv(fname)
    llm_decisions = get_gpt4_score(df["Question"].tolist(), df["Reference"].tolist(), df["Prediction"].tolist())
    llm_decisions = [bool(decisions) for decisions in llm_decisions]
    df["llm_decisions"] = llm_decisions
    df.to_csv(fname, index=False)
    '''



