from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import os
from tqdm import tqdm
from accelerate import cpu_offload
import sys, logging
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, BadRequestError
from huggingface_hub import login
print ("Loading .env was: ", load_dotenv())
n=15
def get_device() -> str:
    # set the device
    if torch.cuda.is_available():
        print("CUDA AVAILABLE....")
        torch.cuda.empty_cache()
        return "cuda"
    else:
        return "cpu"

# Function to generate a Chain of Thought (CoT) prompt
def generate_cot_prompt(question):
    # Define a basic chain-of-thought prompt format
    cot_prompt = f"""
    Question: {question}
    Let's think step by step:
    """
    print (f"CoT Prompt: {cot_prompt}\n")
    return cot_prompt



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


def get_gpt4_score(references:str, predictions:str) -> bool:
    outputs = []
    num_rate_errors = 0
    openai_client = OpenAI(api_key=os.environ.get("HF_API_TOKEN"))
    prompts = [(
        f"You are a Maths Teacher. You will be given the LLM answer to a Maths problem along with the correct answer. Your task is to compare the Generated Answer to the Reference Answer and output True if both answers match and False is the Generated Answer doesn't contain the correct answer as per the Reference Answer"
        f"\n Reference Answer: {ref} \n\n Generated Answer: {pred} \n Output only True or False. Evaluation: ") for ref, pred in zip(references, predictions)]
    for prompt in tqdm(prompts):
        received = False
        while not received:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=10,
                    temperature=0.2,
                )
                received = True
                output = response.choices[0].message.content
                outputs.append(output)

            except:
                outputs.append("")
                error = sys.exc_info()[0]
                num_rate_errors += 1
                if error == BadRequestError:
                    # something is wrong: e.g. prompt too long
                    logging.critical(f"BadRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False
                logging.error("API error: %s (%d)" % (error, num_rate_errors))

    return outputs

if __name__ == '__main__':
    device = get_device()
    login(os.environ.get("HF_API_TOKEN"),add_to_git_credential = True)


    # Load the GSM8k dataset from Hugging Face
    dataset = load_dataset("openai/gsm8k", "main", split='test')
    # Test on a subset from the GSM8k dataset
    # Load the Llama 3 8B model and tokenizer from Hugging Face
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                 torch_dtype=torch.bfloat16)  # .to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

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
    df.to_csv("llama3_gsm8k.csv",index=False)

    #decision = get_gpt4_score(reference, prediction)

