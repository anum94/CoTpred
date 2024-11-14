import os

import pandas as pd
import regex as re
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

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
            model="gpt-4o",
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

    df = pd.read_excel("/Users/anumafzal/PycharmProjects/ToTpred/runs/deepmind-aqua_rat/2024-11-11_22-51-20/CoT_True/deepmind-aqua_rat.xlsx")


    #df = pd.concat([df1, df2], ignore_index=True)
    questions = list(df["Question"])
    references = list(df["Reference"])
    predictions = list(df["Prediction"])

    llm_output = get_gpt4_score(
        questions=questions, references=references, predictions=predictions
    )
    df["llm_decisions"] = llm_output
    df.to_excel("/Users/anumafzal/PycharmProjects/ToTpred/runs/deepmind-aqua_rat/2024-11-11_22-51-20/CoT_True/deepmind-aqua_rat_gpt_4o.xlsx", index=False)

