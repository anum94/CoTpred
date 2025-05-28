import pandas as pd
import numpy as np
from tqdm import tqdm
def get_pot(question, reference, prediction, llm_label, human_label):
    buckets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    steps = [int(i*len(prediction)) for i in buckets]
    t_steps = []
    prompt = None
    for t, step in enumerate(steps):
        if prompt:
            prompt = question + '\n'  "Let's think step by step:\n " + prediction[:step]
        else:
            prompt = question + "Let's think step by step"

        t_steps.append([prompt, reference,prediction,llm_label,human_label,t])

    df_ = pd.DataFrame(t_steps, columns=['Question', 'Reference', 'Prediction', 'llm_decisions', 'anum_decisions', 't'])

    return df_
'''
print ("Aqua Rag Dataset")
input_file = "../cot_true_with_options_9666_aqua_rag/t0/deepmind-aqua_rat_gpt_4o_balanced_13524_balanced_9666.xlsx"
input_file = ("../cot_true_with_options_9666_aqua_rag/t0/test_set/deepmind-aqua_rat_balanced_1000_labelled.xlsx")
df = pd.read_excel(input_file)
per_sample_pots = []
for question, reference, prediction, llm_label, human_label in tqdm(zip(df['Question'], df['Reference'], df['Prediction'],
                                                                   df['llm_decisions'], df['anum_decisions']), total= len(df)):
    per_sample_pot = get_pot(question, reference, prediction, llm_label, human_label)
    per_sample_pots.append(per_sample_pot)

final_df = pd.concat(per_sample_pots)
output_file = f"../cot_true_with_options_9666_aqua_rag/prediction_over_time/deepmind-aqua_rat_balanced_9666_PoT_{len(final_df)}.xlsx"
output_file = f"../cot_true_with_options_9666_aqua_rag/prediction_over_time/test_set/deepmind-aqua_rat_balanced_1000_labelled_PoT_{len(final_df)}.xlsx"
final_df.to_excel(output_file, index = False)
print (len(final_df))
print(final_df.groupby(['t']).size())
steps_to_use = [0,3,8,12,20,30]
final_df = final_df.loc[final_df['t'].isin(steps_to_use)]
print (len(final_df))
output_file = f"../cot_true_with_options_9666_aqua_rag/prediction_over_time/deepmind-aqua_rat_gpt_4o_balanced_balanced_9666_PoT{steps_to_use}_{len(final_df)}.xlsx"
output_file = f"../cot_true_with_options_9666_aqua_rag/prediction_over_time/test_set/deepmind-aqua_rat_balanced_1000_labelled_PoT{steps_to_use}_{len(final_df)}.xlsx"
final_df.to_excel(output_file, index = False)

print ("cnk12 Dataset")
input_file = "../cn-k12/T_0/aslawliet-cn-k12_train_12160.xlsx"
#input_file = "../cn-k12/T_0/aslawliet-cn-k12_test_1000_labeled.xlsx"
df = pd.read_excel(input_file).head(9900)
per_sample_pots = []
for question, reference, prediction, llm_label, human_label in tqdm(zip(df['Question'], df['Reference'], df['Prediction'],
                                                                   df['llm_decisions'], df['anum_decisions']), total= len(df)):
    per_sample_pot = get_pot(question, reference, prediction, llm_label, human_label)
    per_sample_pots.append(per_sample_pot)

final_df = pd.concat(per_sample_pots)
output_file = f"../cn-k12/prediction_over_time/aslawliet-cn-k12_train_12160_PoT_{len(final_df)}.xlsx"
#output_file = f"../cn-k12/prediction_over_time/aslawliet-cn-k12_test_1000_PoT_{len(final_df)}.xlsx"
final_df.to_excel(output_file, index = False)
print (len(final_df))
print(final_df.groupby(['t']).size())

'''

print ("Olympiad Dataset")
input_file = "../datasets/llama/olympiad/T_0/olympiad_train_9996.xlsx"
input_file = "../datasets/llama/olympiad/T_0/olympiad_test_1000_labeled.xlsx"
df = pd.read_excel(input_file)
per_sample_pots = []
for question, reference, prediction, llm_label, human_label in tqdm(zip(df['Question'], df['Reference'], df['Prediction'],
                                                                   df['llm_decisions'], df['anum_decisions']), total= len(df)):
    per_sample_pot = get_pot(question, reference, prediction, llm_label, human_label)
    per_sample_pots.append(per_sample_pot)

final_df = pd.concat(per_sample_pots)
output_file = f"../datasets/llama/olympiad/Pred_over_time/olympiad_train_9996_PoT_{len(final_df)}.xlsx"
output_file = f"../datasets/llama/olympiad/Pred_over_time/olympiad_test_1000_labeled_PoT_{len(final_df)}.xlsx"
final_df.to_excel(output_file, index = False)
print (len(final_df))
print(final_df.groupby(['t']).size())
