import pandas as pd
import numpy as np
from tqdm import tqdm
def get_pot(question, reference, prediction, llm_label, human_label):
    steps = [x for x in prediction.split('\n') if x]
    t_steps = []
    completition = None
    for t, step in enumerate(steps):
        if completition:
            completition = completition + '\n' + step
            prompt = question + completition
        else:
            prompt = question + "Let's think step by step"
            completition = ' '
        t_steps.append([prompt, reference,prediction,llm_label,human_label,t])

    df_ = pd.DataFrame(t_steps, columns=['Question', 'Reference', 'Prediction', 'llm_decisions', 'anum_decisions', 't'])

    return df_
input_file = "../cot_true_with_options_9666_aqua_rag/t0/deepmind-aqua_rat_gpt_4o_balanced_13524_balanced_9666.xlsx"

df = pd.read_excel(input_file)
per_sample_pots = []
for question, reference, prediction, llm_label, human_label in tqdm(zip(df['Question'], df['Reference'], df['Prediction'],
                                                                   df['llm_decisions'], df['anum_decisions']), total= len(df)):
    per_sample_pot = get_pot(question, reference, prediction, llm_label, human_label)
    per_sample_pots.append(per_sample_pot)

final_df = pd.concat(per_sample_pots)
output_file = f"../cot_true_with_options_9666_aqua_rag/prediction_over_time/deepmind-aqua_rat_gpt_4o_balanced_balanced_9666_PoT_{len(final_df)}.xlsx"
final_df.to_excel(output_file, index = False)
print (len(final_df))
print(final_df.groupby(['t']).size())
steps_to_use = [0,3,8,12,20,30]
final_df = final_df.loc[final_df['t'].isin(steps_to_use)]
print (len(final_df))
output_file = f"../cot_true_with_options_9666_aqua_rag/prediction_over_time/deepmind-aqua_rat_gpt_4o_balanced_balanced_9666_PoT{steps_to_use}_{len(final_df)}.xlsx"
final_df.to_excel(output_file, index = False)
