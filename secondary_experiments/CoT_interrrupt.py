import pandas as pd
from together import Together
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm
from utils.classify_math import get_gpt4_score
def get_device() -> str:
    # set the device
    if torch.cuda.is_available():
        print("CUDA AVAILABLE....")
        torch.cuda.empty_cache()
        return "cuda"
    else:
        return "cpu"

device = get_device()
def generate_answer(input, togetherai = True):
    if togetherai == False:
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                     torch_dtype=torch.bfloat16, output_hidden_states=True,
                                                     return_dict_in_generate=True,
                                                     # load_in_8bit=True
                                                     )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = input.split("Let's think step by step:")
    messages = [
           {"role": "user", "content": inputs[0] + "Let's think step by step:"},
        {"role": "assistant",
            "content": inputs[1]},
            {"role": "user", "content": "\n Stop all computation and get give me the correct answer in 2- 3 words, if you know it already. Answer: "}
        ]
    if togetherai:
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY")) #'95d4cde2369d3c193c5ad57d5efbad3d3dbe77250d458bc304834bd43b65c037'
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=messages,
            max_tokens=15, #tokens to generate
            temperature=0,
            top_p=1,
            do_sample=False,
        )
        answer = response.choices[0].message.content
        #print("OUTPUT:", answer)
    else:
        # Tokenize the input prompt
        inputs = tokenizer(messages, return_tensors="pt", padding=True,truncation=True) #.to(device)
        #cpu_offload(model, device, offload_buffers=True)
        inputs = inputs.to(device)

        # Generate output from the model (limiting max tokens for efficiency)
        outputs = model.generate(
            **inputs,
            max_length=15,  # Adjust this to limit the length of the response
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

def generate_answer_no_cot(input, togetherai = True):
    if togetherai == False:
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                     torch_dtype=torch.bfloat16, output_hidden_states=True,
                                                     return_dict_in_generate=True,
                                                     # load_in_8bit=True
                                                     )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = input.split("Let's think step by st")
    prompt = (f"Generate answer of the given question without any chain of thought prompting. "
                   f"Do not give any reasoning or explanations as a part of the output. \n"
                   f"Question: {inputs[0]} \n Please generate the answer in few words without any explanations.")
    messages = [
           {"role": "user", "content": prompt},
        ]
    if togetherai:
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY")) #'95d4cde2369d3c193c5ad57d5efbad3d3dbe77250d458bc304834bd43b65c037'
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=messages,
            max_tokens=15, #tokens to generate
            temperature=0,
            top_p=1,
            do_sample=False,
        )
        answer = response.choices[0].message.content
        #print("OUTPUT:", answer)
    else:
        # Tokenize the input prompt
        inputs = tokenizer(messages, return_tensors="pt", padding=True,truncation=True) #.to(device)
        #cpu_offload(model, device, offload_buffers=True)
        inputs = inputs.to(device)

        # Generate output from the model (limiting max tokens for efficiency)
        outputs = model.generate(
            **inputs,
            max_length=15,  # Adjust this to limit the length of the response
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

def interupt_generation(path, steps, n = 100, ds = "ds"):
    df = pd.read_excel(path, header=0)
    df = (df.loc[df['t'].isin(steps)]).head(n)
    generations = []
    for question in tqdm(df["Question"]):
        generation = generate_answer(question, togetherai=True)
        generations.append(generation)
    df['interrupted_gen'] = generations
    new_path = os.path.join(os.path.dirname(path), f"{ds}_gen_interupt_steps.xlsx")
    print (new_path)
    df.to_excel(new_path)
def without_CoT(path, steps, n = 100, ds = "ds"):
    df = pd.read_excel(path, header=0)
    df = df.loc[df['t']==0].head(n)
    generations = []
    for question in tqdm(df["Question"]):
        generation = generate_answer_no_cot(question, togetherai=True)
        generations.append(generation)
    df['without_cot_gen'] = generations
    print ("Evaluating")
    df["llm_decisions_without_cot_gen"] = get_gpt4_score(df["Question"].tolist(), df["Reference"].tolist(), df['without_cot_gen'] .tolist())
    new_path = os.path.join(os.path.dirname(path), f"{ds}_gen_without_CoT_100.xlsx")
    print(new_path)
    print(f"Overall Correct without CoT: {df['llm_decisions_without_cot_gen'].value_counts()[1]}")
    df.to_excel(new_path)

def summarize_annotations(filename, steps):
    df = pd.read_excel(filename)
    q = [question.split("Let's think step")[0] for question in df["Question"].tolist()]
    df["interrupted_gen_llm_decisions"] = get_gpt4_score(q, df["Reference"].tolist(),
                                                         df['interrupted_gen'].tolist())

    df.to_excel(filename)
    df1 = (df.loc[df['t'] == steps[0]])
    df2 = (df.loc[df['t'] == steps[1]])
    df3 = (df.loc[df['t'] == steps[2]])

    print(f"{filename} \n")
    count = df1['interrupted_gen_label'].value_counts().to_dict()
    print (f"step: {steps[0]} \n {count}")
    print(f"Overall Correct: {df1['interrupted_gen_llm_decisions'].value_counts()[1]}")


    count = df2['interrupted_gen_label'].value_counts().to_dict()
    print(f"step: {steps[1]} \n {count}")
    print(f"Overall Correct: {df2['interrupted_gen_llm_decisions'].value_counts()[1]}")

    count = df3['interrupted_gen_label'].value_counts().to_dict()
    print(f"step: {steps[2]} \n {count}")
    print(f"Overall Correct: {df3['interrupted_gen_llm_decisions'].value_counts()[1]}")


aqua_t = (4, 6 , 9)
olympiad_t = (2, 4, 9)
cnk12_t = ( 3, 5,9)

aqua_pot_path = "datasets/llama/cot_true_with_options_9666_aqua_rag/prediction_over_time/deepmind-aqua_rat_balanced_1000_labelled_PoT_10000.xlsx"
olympiad_pot_path = "../datasets/llama/olympiad/pred_over_time/olympiad_test_1000_labeled_PoT_10000.xlsx"
cnk12_pot_path = "../datasets/llama/cn-k12/prediction_over_time/aslawliet-cn-k12_test_1000_PoT_10000.xlsx"
n = 100
#interupt_generation(aqua_pot_path, steps=aqua_t, n = n, ds="aqua")
#without_CoT(aqua_pot_path, steps=aqua_t, n = n, ds="aqua")
#without_CoT(olympiad_pot_path, steps=olympiad_t, n = n, ds="aqua")
#without_CoT(cnk12_pot_path, steps=cnk12_t, n = n, ds="aqua")
#interupt_generation(olympiad_pot_path, steps=olympiad_t, n = n, ds="olympiad")
#interupt_generation(cnk12_pot_path, steps=cnk12_t, n = n, ds = "cnk12")

# analyze results
aqua = "cot_true_with_options_9666_aqua_rag/prediction_over_time/aqua_gen_interupt_steps_marko_annotated.xlsx"
summarize_annotations(aqua, aqua_t)

olympiad = "olympiad/pred_over_time/olympiad_gen_interupt_steps_annotated_nico.xlsx"
summarize_annotations(olympiad, olympiad_t)

cnk12 = "cn-k12/prediction_over_time/cnk12_gen_interupt_steps_final_labeled_ishwor.xlsx"
summarize_annotations(cnk12, cnk12_t)

print ()