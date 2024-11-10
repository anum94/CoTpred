import pandas as pd

def compare(cot_path,no_cot_path):

    df_no_cot = pd.read_excel(no_cot_path)
    df_no_cot = df_no_cot.drop(['anum_decisions'], axis=1)
    df_no_cot.columns = ['Question', 'Reference', 'Prediction_no_cot','llm_decisions_no_cot']


    df_cot = pd.read_excel(cot_path)
    df_cot = df_cot.drop(['anum_decisions', 'Question', 'Reference'], axis=1)
    df_cot.columns = [ 'Prediction_cot','llm_decisions_cot']

    df = pd.concat([df_no_cot, df_cot], axis=1)
    agreements = [True if a==b else False for a,b in zip(df['llm_decisions_no_cot'], df['llm_decisions_cot'])]
    return df , agreements

gsm8k_cot_path = "../runs/processed_ds/gsm8k/CoT_True/openai-gsm8k_train_test_cot.xlsx"
gsm8k_no_cot_path = "../runs/processed_ds/gsm8k/CoT_False/openai-gsm8k_no_cot_train_test.xlsx"
gsm8k_df, gsm8k_agreements = compare(cot_path = gsm8k_cot_path, no_cot_path = gsm8k_no_cot_path)

print ("gsm8k:")
print("Agreement: ",gsm8k_agreements.count(True))
print("Disagreement: ",gsm8k_agreements.count(False))


aqua_rat_no_cot_path = "../runs/processed_ds/deepmind-aqua_rat/CoT_False/deepmind-aqua_rat.xlsx"
aqua_rat_cot_path = "../runs/processed_ds/deepmind-aqua_rat/CoT_True/deepmind-aqua_rat.xlsx"
aqua_rat_df , aqua_rat_agreements = compare(cot_path = aqua_rat_cot_path, no_cot_path = aqua_rat_no_cot_path)

print ("Aqua Rat:")
print("Agreement: ",aqua_rat_agreements.count(True))
print("Disagreement: ",aqua_rat_agreements.count(False))

print ("")