from transformers import AutoTokenizer, AutoModelForCausalLM
from together import Together
import os
from utils.misc import device
# Function to generate a Chain of Thought (CoT) prompt
def generate_cot_prompt(questions):
    # Define a basic chain-of-thought prompt format

    if isinstance(questions, str):
        cot_prompts = f"Question: {questions} \nLet's think step by step:\n"
    else:
        cot_prompts = [f"Question: {question} \nLet's think step by step:\n" for question in questions]
    #print (f"CoT Prompt: {cot_prompts[0}\n")
    return cot_prompts

def generate_prompt(questions):
    if isinstance(questions, str):
        cot_prompts = (f"Generate answer of the question in the numberic form without showing intermediate calculation steps."
                       f"\nQuestion: {questions}  \n Answer:\n")
    else:
        cot_prompts = [(f"Generate answer of the given question without any explanation or reasoning."
                    f"Question: {question} \n Please generate the answer in one word. \n Answer:\n") for question in questions]
    #print (f"CoT Prompt: {cot_prompts[0}\n")
    return cot_prompts


# Function to get the model's prediction
def generate_answer(question, tokenizer, CoT, togetherai = True, model = None):
    if togetherai == False and model is None:
        print("Model must be provided to run inference if Together AI is disabled. Model inference Failed. Returning empty string as output")
        return ""
    if CoT:
        cot_prompt = generate_cot_prompt(question)
    else:
        cot_prompt = generate_prompt(question)
    if togetherai:
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[{"role": "user", "content": cot_prompt}],
            max_tokens=10, #tokens to generate
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