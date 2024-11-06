import pandas as pd

df_no_cot = pd.read_excel("../runs/openai-gsm8k/processed_ds/openai-gsm8k_no_cot_train_test.xlsx")
df_no_cot = df_no_cot.drop(['anum_decisions'], axis=1)
df_no_cot.columns = ['Question', 'Reference', 'Prediction_no_cot','llm_decisions_no_cot']

#df = pd.read_excel("../runs/openai-gsm8k/processed_ds/llama3_gsm8k_train_cot.xlsx")
#df2 = pd.read_excel("../runs/openai-gsm8k/processed_ds/openai-gsm8k-test-cot.xlsx")
#df3 = pd.concat([df,df2], axis = 0)
#df3.to_excel("../runs/openai-gsm8k/processed_ds/openai-gsm8k_train_test_cot.xlsx")

df_cot = pd.read_excel("../runs/openai-gsm8k/processed_ds/openai-gsm8k_train_test_cot.xlsx")
df_cot = df_cot.drop(['anum_decisions', 'Question', 'Reference'], axis=1)
df_cot.columns = [ 'Prediction_cot','llm_decisions_cot']

df = pd.concat([df_no_cot, df_cot], axis=1)
agreements = [True if a==b else False for a,b in zip(df['llm_decisions_no_cot'], df['llm_decisions_cot'])]
print("Agreement: ",agreements.count(True))
print("Disagreement: ",agreements.count(False))

