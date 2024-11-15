import pandas as pd
import numpy as np
import os
import torch
import tensorflow as tf
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from main import generate_prompt, generate_cot_prompt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

def update_labels(folder_path, decisions):

    fname = os.path.join(folder_path, "regression_labels_labelled.txt")
    decisions = [str(d) for d in decisions]
    np.savetxt(fname, np.array(decisions), fmt='%d')
    print(f"Saved Regression Labels at {fname}")
    return fname

def contruct_regression_features(df, folder_path, CoT):
    def get_last_token_idx(inputs_ids: list) -> list:
        last_token_idx = list()
        for input_id in inputs_ids:
            if tokenizer.pad_token_id in input_id:
                idx = input_id.index(tokenizer.pad_token_id) - 1
            else:
                idx = -1
            last_token_idx.append(idx)

        return last_token_idx
    input_sentence = list(df['Question'])
    if CoT:
        input_prompts = generate_cot_prompt(input_sentence)
    else:
        input_prompts = generate_prompt(input_sentence)

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                 torch_dtype=torch.bfloat16, output_hidden_states=True,
                                                 return_dict_in_generate=True,
                                                 # load_in_8bit=True
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(input_prompts, padding=True, truncation=True, return_tensors="pt")
    last_token_indices = get_last_token_idx(inputs['input_ids'].tolist())
    batch_size = 4

    # Run forward pass with a batch size of 2
    # Ensure inputs are divided as per batch size
    print("Generating Features of regression model")
    input_ids_batches = inputs['input_ids'].split(batch_size)
    attention_mask_batches = inputs['attention_mask'].split(batch_size)

    # Process each batch
    feature = None
    for input_ids, attention_mask, last_token_idx in tqdm(
            zip(input_ids_batches, attention_mask_batches, last_token_indices), total=len(input_ids_batches)):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = outputs.hidden_states
        last_layer_hidden_state = hidden_states[-1]
        last_layer_hidden_state = last_layer_hidden_state[:, last_token_idx, :]
        if feature is None:
            feature = last_layer_hidden_state
            # print (feature.size())
        else:
            # print (feature.size(), last_layer_hidden_state.size())
            feature = torch.concat((feature, last_layer_hidden_state), dim=0)

    print(feature.size())
    # X = feature.mean(dim=1)
    # print (X.size())
    feature = feature.float().numpy()
    y = pd.to_numeric(df['llm_decisions_no_cot'])

    fname = os.path.join(folder_path, "regression_features.txt")
    np.savetxt(fname, feature, fmt='%d')
    print(f"Saved Regression Features at {fname}")

    fname = os.path.join(folder_path, "regression_labels.txt")
    np.savetxt(fname, np.array(y), fmt='%d')
    print(f"Saved Regression Labels at {fname}")

    return feature, y
def compute_metrics(predictions, true_labels, pred_prob):

    # Calculate Precision
    precision = precision_score(true_labels, predictions)
    print(f'Precision: {precision}')

    # Calculate Recall
    recall = recall_score(true_labels, predictions)
    print(f'Recall: {recall}')

    # Calculate F1 Score
    f1 = f1_score(true_labels, predictions)
    print(f'F1 Score: {f1}')

    # Calculate Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predictions)
    print(f'Confusion Matrix:\n{conf_matrix}')

    # Calculate AUC
    auc = roc_auc_score(true_labels, pred_prob)
    print(f'Area Under Curve (AUC): {auc}')
def evaluate_model(model_path, feature_path, label_path, n = None, outfile = None, decisions = None):

    X_test = np.loadtxt(feature_path, dtype=int)
    if decisions is None:
        y_test = np.loadtxt(label_path, dtype=int)
    else:

        y_test = np.array(decisions, dtype=int)
    if n is not None:
        X_test = X_test[:n]
        y_test = y_test[:n]
    best_model = tf.keras.models.load_model(model_path)

    # Evaluate the model
    loss, accuracy = best_model.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy:.4f}')

    # Get predictions
    log_prob = best_model.predict(X_test, verbose=1)
    pred = (log_prob > 0.5).astype("int32")

    compute_metrics(predictions=pred, true_labels=y_test, pred_prob = log_prob)
    agreements = [True if a == b else False for a, b in zip(pred, y_test)]
    results = pd.DataFrame()
    results['agreements'] = agreements
    if outfile is not None:
        dir =  os.path.dirname(model_path)

        if n is not None:
            file = os.path.join(dir, str(n),outfile)
        else:
            file = os.path.join(dir, outfile)

        results.to_excel(file)
        print (file)


    return accuracy , loss


'''
test_no_cot_path= "/Users/anumafzal/PycharmProjects/ToTpred/runs/processed_ds/deepmind-aqua_rat/test_set/CoT_False/deepmind-aqua_rat_6k.xlsx"
test_cot_path = "/Users/anumafzal/PycharmProjects/ToTpred/runs/processed_ds/deepmind-aqua_rat/test_set/CoT_True/deepmind-aqua_rat_6k.xlsx"


df_no_cot = pd.read_excel(test_no_cot_path)
#df_no_cot = df_no_cot.drop(['anum_decisions'], axis=1)
df_no_cot.columns = ['Question', 'Reference', 'Prediction_no_cot','anum_decisions_no_cot','llm_decisions_no_cot']


# Fot No CoT, get ids of all correct and all incorrect
no_cot_true_index = df_no_cot.index[df_no_cot['llm_decisions_no_cot'] == 1].tolist()
no_cot_false_index = df_no_cot.index[df_no_cot['llm_decisions_no_cot'] == 0].tolist()

df_cot = pd.read_excel(test_cot_path)
df_cot = df_cot.drop([ 'Question', 'Reference'], axis=1)
df_cot.columns = [ 'Prediction_cot', 'anum_decisions_cot','llm_decisions_cot']

df = pd.concat([df_no_cot, df_cot], axis=1)
agreements = [True if a==b else False for a,b in zip(df['llm_decisions_no_cot'], df['llm_decisions_cot'])]
cot_true_no_cot_true_index = df.index[(df['llm_decisions_cot'] == 1) & (df['llm_decisions_no_cot'] == 1)].tolist()
cot_false_no_cot_false_index = df.index[(df['llm_decisions_cot'] == 0) & (df['llm_decisions_no_cot'] == 0)].tolist()

print("Both can solve: ",len(cot_true_no_cot_true_index), f" ({len(cot_true_no_cot_true_index)*100/len(df)}%)")
print("both cannot solve: ",len(cot_false_no_cot_false_index), f" ({len(cot_false_no_cot_false_index)*100/len(df)})%")


cot_true_no_cot_false_index = df.index[(df['llm_decisions_cot'] == 1) & (df['llm_decisions_no_cot'] == 0)].tolist()
cot_false_no_cot_true_index = df.index[(df['llm_decisions_cot'] == 0) & (df['llm_decisions_no_cot'] == 1)].tolist()
print("CoT can and No Cot can solve: ", len(cot_true_no_cot_false_index), f" ({len(cot_true_no_cot_false_index)*100/len(df)}%)")
print("CoT cannot  and No Cot can solve: ", len(cot_false_no_cot_true_index), f" ({len(cot_false_no_cot_true_index)*100/len(df)}%)")


# Fot CoT, get ids of all correct and all incorrect
cot_true_index = df_cot.index[df_cot['llm_decisions_cot'] == 1].tolist()
cot_false_index = df_cot.index[df_cot['llm_decisions_cot'] == 0].tolist()

common_false_index = list(set(cot_false_index) & set(no_cot_false_index))
common_true_index = list(set(cot_true_index) & set(no_cot_true_index))

if len(common_false_index) > len(common_true_index):
    n = len(common_true_index)
    common_false_index = common_false_index[:n]
elif len(common_true_index) > len(common_false_index):
    n = len(common_false_index)
    common_true_index = common_true_index[:n]

balanced_test_set_index = common_true_index +common_false_index

df = df.iloc[balanced_test_set_index]
print (f"balance test set has {len(df)} samples.")
agreements = [True if a==b else False for a,b in zip(df['llm_decisions_no_cot'], df['llm_decisions_cot'])]

df = df.sample(frac=1)
#df.to_excel(f"/Users/anumafzal/PycharmProjects/ToTpred/runs/processed_ds/deepmind-aqua_rat/test_set/balanced_{len(df)}_6k.xlsx")

CoT = False

df = pd.read_excel("runs/processed_ds/deepmind-aqua_rat/test_set/balanced_1044_6k.xlsx")

contruct_regression_features(df, folder_path, CoT)




'''
CoT = True
folder_path = f"runs/processed_ds/deepmind-aqua_rat/test_set/CoT_{CoT}/"
df = pd.read_excel("runs/processed_ds/deepmind-aqua_rat/test_set/balanced_1044_6k_200_labelled.xlsx")
#label_path = update_labels(folder_path, decisions=list(df["anum_decisions_cot"]))
labels = list(df["anum_decisions_cot"])

feature_path = "runs/processed_ds/deepmind-aqua_rat/test_set/CoT_False/regression_features.txt"
label_path = "runs/processed_ds/deepmind-aqua_rat/test_set/CoT_False/regression_labels.txt"
model_path = "runs/processed_ds/deepmind-aqua_rat/without_options/CoT_False/best_model.keras"
outfile = f"CoT_{CoT}.xlsx"
evaluate_model(model_path, feature_path, label_path, outfile=outfile, n = 200, decisions=list(df["anum_decisions_cot"]))
