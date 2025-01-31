import os

import pandas as pd
import regex as re
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer
load_dotenv()


def check_class_imbalance(df: pd.DataFrame):

    true_label = len(df[df["llm_decisions"] == 1])
    false_label = len(df[df["llm_decisions"] == 0])
    print (f" True Labels: {true_label}, False Labels: {false_label}")
    print (f"LLM can generate correct answer for {(true_label / (true_label + false_label))*100}% of the samples")
    return true_label, false_label

def get_balanced_ds(df, samples_per_class):


    df_false = df[df["llm_decisions"] == 0].head(samples_per_class)
    df_true = df[df["llm_decisions"] == 1].head(samples_per_class)

    df = pd.concat([df_true, df_false], ignore_index=True)
    print(f"Using only {len(df)} samples to fix class imbalance in the dataset.")
    df = df.sample(frac=1)

    return df
def extract_number(text):
    try:
        # print("TEXT:", text)
        # Use regex to find all numbers in the string
        numbers = re.findall(r"\d+", text)
        # Convert the list of number strings to integers
        # print(numbers)
        return float(numbers[0])
    except IndexError:
        return float(-1)
def drop_nasty_samples(df):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    good_index = []
    print ("Remove bad samples after inferences.")
    for i, input in tqdm(enumerate(list(df['Prediction']))):

        token = tokenizer(input, return_tensors="pt")
        token = token['input_ids']
        num_tokens = token.shape[1]
        if num_tokens <= 512:
            good_index.append(i)
    len_before = len(df)
    df = df.iloc[good_index]
    len_after = len(df)
    print (f"Dropped {len_before-len_after} samples.")
    return df

def get_gpt4_score(questions: list, references: list, predictions: list) -> list:
    outputs = []
    num_rate_errors = 0
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))
    prompts = [
        (
            f"You are a Maths Teacher. You will be given the LLM answer to a Maths or Reasoning Question along with the correct answer. Your task is to compare the Generated Answer to the Reference Answer for the given question. "
            f"You should output 1 if generated answer contains has the correct output in the end, and 0 is the Generated Answer doesn't contain the correct answer as per the Reference Answer"
            f"\n Question: {ques}"
            f"\n Reference Answer: {ref} \n\n Generated Answer: {pred} \n Output only 0 or 1 depending on the Evaluation: "
        )
        for ques, ref, pred in zip(questions, references, predictions)
    ]
    for prompt in tqdm(prompts):
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=10,
            temperature=0.2,
        )

        output = response.choices[0].message.content
        outputs.append(extract_number(output))

    return outputs


if __name__ == "__main__":

    df = pd.concat([pd.read_excel("../olympiad/olympiad_train_9902.xlsx"),
        pd.read_excel("../olympiad/olympiad_train_94.xlsx"),


    ])
    new_fname = f"../olympiad/olympiad_train_{len(df)}.xlsx"
    df.to_excel(new_fname, index=False)

    df= df[df['Prediction'].notna()]
    #df['Prediction'] = [pred.split("Let's think step by step:")[1] for pred in df['Prediction']]
    df = drop_nasty_samples(df)
    df.to_excel(f"../olympiad/olympiad_49-57k_cleaned.xlsx", index=False)
    questions = list(df["Question"])
    references = list(df["Reference"])
    predictions = list(df["Prediction"])

    llm_output = get_gpt4_score(
        questions=questions, references=references, predictions=predictions
    )
    df["llm_decisions"] = llm_output

    df.to_excel(f"../olympiad/olympiad_49-57k_llm_labelled{len(df)}.xlsx", index=False)

    n_true_label, n_false_label = check_class_imbalance(df)
    samples_per_class = min(n_false_label, n_true_label)
    df = get_balanced_ds(df, samples_per_class=samples_per_class)
    new_fname = f"../olympiad/olympiad_train_{len(df)}.xlsx"
    df.to_excel(new_fname, index=False)

