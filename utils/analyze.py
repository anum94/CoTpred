import pandas as pd

def get_coorelation(filename, n = 100):
    print (filename)
    df = pd.read_excel(filename)

    true_label = len(df[df["llm_decisions"] == 1])
    false_label = len(df[df["llm_decisions"] == 0])
    print(f" True Labels: {true_label}, False Labels: {false_label}")
    print(f"LLM can generate correct answer for {(true_label / (true_label + false_label)) * 100}% of the samples")

    if len(df) > 100:
        df = df.head(n)
    print(f"N: {len(df)}")
    agreements = [True if a == b else False for a, b in zip(df['llm_decisions'], df['anum_decisions'])]
    rate  = agreements.count(True) / len (agreements)
    print (rate)
    return rate

#from create_balanced_test_set import compute_metrics

'''
df_no_cot = pd.read_excel("../runs/processed_ds/deepmind-aqua_rat/without_options/CoT_False/deepmind-aqua_rat_25k.xlsx")
df_no_cot = df_no_cot.drop(['anum_decisions'], axis=1)
df_no_cot.columns = ['Question', 'Reference', 'Prediction_no_cot','llm_decisions_no_cot']


df_cot = pd.read_excel("../runs/processed_ds/deepmind-aqua_rat/without_options/CoT_True/deepmind-aqua_rat_gpt_4o.xlsx")
df_cot = df_cot.drop(['anum_decisions', 'Question', 'Reference'], axis=1)
df_cot.columns = [ 'Prediction_cot','llm_decisions_cot']
'''


filenames = ["../test_files/lighteval-MATH_final.xlsx",
            "../test_files/aslawliet-cn-k12_final.xlsx",
            "../test_files/aops_forum_final.xlsx",
            "../test_files/aops_forum_filtered_balanced_66_final.xlsx"
            ]

for filename in filenames:
    get_coorelation(filename)
'''

df_cot = pd.read_excel("../runs/processed_ds/deepmind-aqua_rat/without_options/CoT_True/CoT_True.xlsx")
df_no_cot = pd.read_excel("../runs/processed_ds/deepmind-aqua_rat/without_options/CoT_False/CoT_False.xlsx")
df = pd.concat([df_no_cot, df_cot], axis=1)
df.columns = ['num', 'no_cot', 'num', 'cot']
print ("# samples: ",len(df))
agreements = [True if a==b else False for a,b in zip(df['no_cot'], df['cot'])]
cot_true_no_cot_true_index = df.index[(df['cot'] == 1) & (df['no_cot'] == 1)].tolist()
cot_false_no_cot_false_index = df.index[(df['cot'] == 0) & (df['no_cot'] == 0)].tolist()

print("Both pred correct: ",len(cot_true_no_cot_true_index), f" ({len(cot_true_no_cot_true_index)*100/len(df)}%)")
print("both pred wrong: ",len(cot_false_no_cot_false_index), f" ({len(cot_false_no_cot_false_index)*100/len(df)})%")

cot_true_no_cot_false_index = df.index[(df['cot'] == 1) & (df['no_cot'] == 0)].tolist()
cot_false_no_cot_true_index = df.index[(df['cot'] == 0) & (df['no_cot'] == 1)].tolist()
print("CoT can apredict correct, No Cot cannot predict: ", len(cot_true_no_cot_false_index), f" ({len(cot_true_no_cot_false_index)*100/len(df)}%)")
print("Both CoT and no Cot cannot predict: ", len(cot_false_no_cot_true_index), f" ({len(cot_false_no_cot_true_index)*100/len(df)}%)")

'''