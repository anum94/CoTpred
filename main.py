from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import os
from tqdm import tqdm
import sys, logging
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, BadRequestError
from huggingface_hub import login
print ("Loading .env was: ", load_dotenv())
n=15
batch_size = 2
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

if __name__ == '__main__':


    device = get_device()
    login(os.environ.get("HF_API_TOKEN"),add_to_git_credential = True)


    # Load the GSM8k dataset from Hugging Face
    dataset = load_dataset("openai/gsm8k", "main", split='test')
    # Test on a subset from the GSM8k dataset
    # Load the Llama 3 8B model and tokenizer from Hugging Face
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    fname = "llama3_gsm8k.csv"

    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                 torch_dtype=torch.bfloat16, output_hidden_states=True, load_in_4bit=True)  # .to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    '''
    dummy = list()
    for i in tqdm(range(n)):
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
    df.to_csv(fname,index=False)
    
    tokenizer = None
    model = None
    df = pd.read_csv(fname)
    llm_decisions = get_gpt4_score(df["Question"].tolist(), df["Reference"].tolist(), df["Prediction"].tolist())
    llm_decisions = [bool(decisions) for decisions in llm_decisions]
    df["llm_decisions"] = llm_decisions
    df.to_csv(fname, index=False)
    '''

    #tokenizer = None
    #model = None
    fname = "llama3_gsm8k.xlsx"
    feature = None
    df = pd.read_excel(fname, )
    df.columns = ['Question', 'Reference', 'Prediction', 'llm_decisions', 'anum_decisions']
    df_correct = df[df['anum_decisions'] == 1]
    df_incorrect = df[df['anum_decisions'] == 0]

    input_sentence = list(df_correct['Question'])
    input_prompts = generate_cot_prompt(input_sentence)

    inputs = tokenizer(input_prompts, padding=True, truncation=True, return_tensors="pt")

    # Run forward pass with a batch size of 2
    # Ensure inputs are divided as per batch size
    input_ids_batches = inputs['input_ids'].split(batch_size)
    attention_mask_batches = inputs['attention_mask'].split(batch_size)


    # Process each batch
    for input_ids, attention_mask in zip(input_ids_batches, attention_mask_batches):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)


        #outputs = model(**inputs)
        hidden_states = outputs.hidden_states
        last_layer_hidden_state = hidden_states[-1]
        if feature is None:
            feature = last_layer_hidden_state
            print (feature.size())
        else:
            print (feature.size(), last_layer_hidden_state.size())
            feature = torch.concat((feature,last_layer_hidden_state),dim=0)


        print(last_layer_hidden_state.shape)
    print (feature.size())
    avg_features = feature.mean(dim=1)
    print (avg_features.size())


