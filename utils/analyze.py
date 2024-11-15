import pandas as pd

#from create_balanced_test_set import compute_metrics

'''
df_no_cot = pd.read_excel("../runs/processed_ds/deepmind-aqua_rat/without_options/CoT_False/deepmind-aqua_rat_25k.xlsx")
df_no_cot = df_no_cot.drop(['anum_decisions'], axis=1)
df_no_cot.columns = ['Question', 'Reference', 'Prediction_no_cot','llm_decisions_no_cot']


df_cot = pd.read_excel("../runs/processed_ds/deepmind-aqua_rat/without_options/CoT_True/deepmind-aqua_rat_gpt_4o.xlsx")
df_cot = df_cot.drop(['anum_decisions', 'Question', 'Reference'], axis=1)
df_cot.columns = [ 'Prediction_cot','llm_decisions_cot']
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

