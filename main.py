from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import os
from accelerate import cpu_offload

from dotenv import load_dotenv
from huggingface_hub import login
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
    #cpu_offload(model, 'gpu', offload_buffers=True)

    # Generate output from the model (limiting max tokens for efficiency)
    outputs = model.generate(
        **inputs,
        max_length=512,  # Adjust this to limit the length of the response
        num_beams=1,  # Beam search to improve output quality
        #early_stopping=True
    )

    # Decode the model's output
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the final answer after the step-by-step reasoning
    return answer


def get_gpt4_score(reference:str, prediction:str) -> Bool:
    return True

if __name__ == '__main__':
    device = get_device()
    login(os.environ.get("HF_API_TOKEN"),add_to_git_credential = True)


    # Load the GSM8k dataset from Hugging Face
    dataset = load_dataset("openai/gsm8k", "main", split='test')
    # Test on one example from the GSM8k dataset
    sample = dataset[0]
    question = sample['question']
    answer = sample['answer']

    # Load the Llama 3 8B model and tokenizer from Hugging Face
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                 torch_dtype=torch.bfloat16) #.to(device)
    gen_answer = generate_answer(question)

    # Display the question and model's chain-of-thought response
    print(f"Question: {question}")
    print(f"Model's Answer with Chain of Thought:\n{gen_answer}")
    print(f"Reference Answer:\n{answer}")
    dummy = [question, answer, gen_answer]

    #decision = get_gpt4_score(reference, prediction)

