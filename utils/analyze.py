import pandas as pd

df_no_cot = pd.read_excel("../runs/processed_ds/gsm8k/CoT_False/openai-gsm8k_no_cot_train_test.xlsx")
df_no_cot = df_no_cot.drop(['anum_decisions'], axis=1)
df_no_cot.columns = ['Question', 'Reference', 'Prediction_no_cot','llm_decisions_no_cot']


df_cot = pd.read_excel("../runs/processed_ds/gsm8k/CoT_True/openai-gsm8k_train_test_cot.xlsx")
df_cot = df_cot.drop(['anum_decisions', 'Question', 'Reference'], axis=1)
df_cot.columns = [ 'Prediction_cot','llm_decisions_cot']

df = pd.concat([df_no_cot, df_cot], axis=1)
agreements = [True if a==b else False for a,b in zip(df['llm_decisions_no_cot'], df['llm_decisions_cot'])]
print("Agreement: ",agreements.count(True), f" ({agreements.count(True)*100/len(df)}%)")
print("Disagreement: ",agreements.count(False), f" ({agreements.count(False)*100/len(df)})%")

cot_true_no_cot_false_index = df.index[(df['llm_decisions_cot'] == 1) & (df['llm_decisions_no_cot'] == 0)].tolist()
cot_false_no_cot_true_index = df.index[(df['llm_decisions_cot'] == 0) & (df['llm_decisions_no_cot'] == 1)].tolist()
print("CoT can and No Cot can solve: ", len(cot_true_no_cot_false_index), f" ({len(cot_true_no_cot_false_index)*100/len(df)}%)")
print("CoT cannot  and No Cot can solve: ", len(cot_false_no_cot_true_index), f" ({len(cot_false_no_cot_true_index)*100/len(df)}%)")
