import pandas as pd
from together import Together
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm
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


def interupt_generation(path, steps, n = 100):
    df = pd.read_excel(aqua_pot_path, header=0)
    df = (df.loc[df['t'].isin(steps)]).head(n)
    generations = []
    for question in tqdm(df["Question"]):
        generation = generate_answer(question, togetherai=False)
        generations.append(generation)
    df['interrupted_gen'] = generations
    new_path = os.path.join(os.path.abspath(path), "gen_interupt.xlsx")
    print (new_path)
    df.to_excel(new_path)



aqua_t = (4 , 6)
olympiad = (1, 3)
cnk12_t = (3 ,5)

aqua_pot_path = "cot_true_with_options_9666_aqua_rag/prediction_over_time/deepmind-aqua_rat_gpt_4o_balanced_balanced_9666_PoT_96660.xlsx"
olympiad_pot_path = "olympiad/pred_over_time/olympiad_train_9996_PoT_99960.xlsx"
cnk12_pot_path = "/Users/anumafzal/Library/Mobile Documents/com~apple~CloudDocs/PyCharm/cn-k12/prediction_over_time/aslawliet-cn-k12_train_12160_PoT_99000.xlsx"

interupt_generation(aqua_pot_path, steps=aqua_t, n = 10)
