from pyexpat import features

import pandas as pd
import numpy as np
import os
from together import Together
import torch
import tensorflow as tf
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BertModel
from main import generate_prompt, generate_cot_prompt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
from sentence_transformers import SentenceTransformer
def update_labels(folder_path, decisions):

    fname = os.path.join(folder_path, "regression_labels_human.txt")
    np.savetxt(fname, decisions, fmt='%d')
    print(f"Saved Regression Labels at {fname}")
    return fname
def contruct_regression_features(df, folder_path, CoT, compute_all = False):
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
    batch_size = 1

    # Run forward pass with a batch size of 2
    # Ensure inputs are divided as per batch size
    print("Generating Features of regression model")
    input_ids_batches = inputs['input_ids'].split(batch_size)
    attention_mask_batches = inputs['attention_mask'].split(batch_size)

    # Process each batch
    features = None
    for input_ids, attention_mask, last_token_idx in tqdm(
            zip(input_ids_batches, attention_mask_batches, last_token_indices), total=len(input_ids_batches)):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = outputs.hidden_states

        if compute_all:  # save for all hidden layers
            hidden_states = torch.stack(list(hidden_states), dim=0)
            last_token_reps = []
            for hidden_state in hidden_states:
                last_token_rep = hidden_state[:, last_token_idx, :]
                last_token_reps.append(last_token_rep)

            if features is None:
                last_token_reps = torch.stack(last_token_reps, dim=0).unsqueeze(0)
            else:
                last_token_reps = torch.stack(last_token_reps, dim=0).unsqueeze(0)

        else:  # just compute for one layer
            last_layer_hidden_state = hidden_states[-1]
            last_token_reps = last_layer_hidden_state[:, last_token_idx, :]

        if features is None:
            features = last_token_reps
        else:
            features = torch.concat((features, last_token_reps), dim=0)

    if compute_all:
        features_temp = []
        features = features.squeeze()
        a, b, c = features.size()
        for i in range(b):
            feature = features[:, i, :]
            feature = feature.float().numpy()
            fname = os.path.join(folder_path, "training_features", f"regression_features_layer_{i}.txt")
            np.savetxt(fname, feature, fmt='%.8f')
            print(f"Saved Regression Features at {fname}")
            features_temp.append(feature)

        features = features_temp

    else:
        fname = os.path.join(folder_path, "training_features", "regression_features.txt")
        features = features.float().numpy()
        np.savetxt(fname, features, fmt='%.8f')
        print(f"Saved Regression Features at {fname}")

    y = pd.to_numeric(df['llm_decisions'])
    fname_label = os.path.join(folder_path, "training_features", "regression_labels.txt")
    np.savetxt(fname, np.array(y), fmt='%d')
    print(f"Saved Regression Labels at {fname_label}")

    return fname, fname_label, features, y
def construct_bert_features(df, folder_path, CoT):
    model_name = "sentence-transformers/msmarco-bert-base-dot-v5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    input_sentence = list(df['Question'])
    if CoT:
        input_prompts = generate_cot_prompt(input_sentence)
    else:
        input_prompts = generate_prompt(input_sentence)

    features = []
    for question in tqdm(input_prompts):
        inputs = tokenizer(question, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        embedding = last_hidden_states[:, 0, :]
        embedding = embedding.squeeze()
        features.append(embedding)

    fname = os.path.join(folder_path, "regression_features.txt")
    np.savetxt(fname, np.array(features), fmt='%.8f')
    print(f"Saved Regression Labels at {fname}")

    y = pd.to_numeric(df['anum_decisions'])
    fname_label = os.path.join(folder_path, "regression_labels.txt")
    np.savetxt(fname_label, np.array(y), fmt='%d')
    print(f"Saved Regression Labels at {fname_label}")

    return fname, fname_label, features, y

def construct_bge_features(df, folder_path, CoT):
    input_sentence = list(df['Question'])
    if CoT:
        input_prompts = generate_cot_prompt(input_sentence)
    else:
        input_prompts = generate_prompt(input_sentence)

    features = []
    for question in tqdm(input_prompts):
        client = Together(
            api_key=os.environ.get("TOGETHER_API_KEY"))
        response = client.embeddings.create(
            model="BAAI/bge-large-en-v1.5",
            input=question
        )
        embedding = response.data[0].embedding
        embedding = torch.Tensor(embedding)
        features.append(embedding)

    fname = os.path.join(folder_path, "regression_features.txt")
    np.savetxt(fname, np.array(features), fmt='%.8f')
    print(f"Saved Regression Labels at {fname}")

    y = pd.to_numeric(df['anum_decisions'])
    fname_label = os.path.join(folder_path, "regression_labels.txt")
    np.savetxt(fname_label, np.array(y), fmt='%d')
    print(f"Saved Regression Labels at {fname_label}")

    return fname, fname_label, features, y

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
def evaluate_model(model_path, features, labels, feature_path = None, label_path = None, n = None, outfile = None,):

    if feature_path is not None:
        X_test = np.loadtxt(feature_path, dtype=float)
    else:
        if isinstance(features, list):
            X_test = torch.stack(features, dim=0)
        else:
            X_test = features
    if label_path is not None:
        y_test = np.loadtxt(label_path, dtype=int)
    else:
        y_test = np.array(labels, dtype=int)

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
    return accuracy , loss
def derived_balances_test_set_basic(fname):
    df = pd.read_excel(fname)
    samples_per_class = 115

    df_false = df[df["llm_decisions"] == 0].head(samples_per_class)
    df_true = df[df["llm_decisions"] == 1].head(samples_per_class)

    df = pd.concat([df_true, df_false], ignore_index=True)
    print(f"Using only {len(df)} samples to fix class imbalance in the dataset.")
    df = df.sample(frac=1)


    new_fname = os.path.basename(fname).split(".xlsx")[0] + f"_balanced_{len(df)}.xlsx"

    #new_fname = os.path.join(os.path.abspath(fname), new_fname)
    df.to_excel('runs/processed_ds/deepmind-aqua_rat/test_set/CoT_True/with_options/deepmind-aqua_rat_balanced_1k_230.xlsx')
    print(f"Saved balaced dataset to {new_fname}")
    return df

def derive_balanced_set_cot():

    test_no_cot_path= "/Users/anumafzal/PycharmProjects/ToTpred/runs/processed_ds/deepmind-aqua_rat/test_set/CoT_False/deepmind-aqua_rat_6k.xlsx"
    test_cot_path = "/runs/processed_ds/deepmind-aqua_rat/test_set/CoT_True/without_options/deepmind-aqua_rat_6k.xlsx"

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
    df.to_excel(f"/Users/anumafzal/PycharmProjects/ToTpred/runs/processed_ds/deepmind-aqua_rat/test_set/balanced_{len(df)}_6k.xlsx")


#contruct_regression_features(df, folder_path, CoT)


#fname = "runs/processed_ds/deepmind-aqua_rat/test_set/CoT_True/with_options/deepmind-aqua_rat_1k.xlsx"
#derived_balances_test_set_basic(fname)


CoT = True
with_options = True

folder_path = f"/Users/anumafzal/PycharmProjects/ToTpred/runs/processed_ds/deepmind-aqua_rat/test_set/CoT_{CoT}/with_options/"
df_marko = pd.read_excel("/Users/anumafzal/PycharmProjects/ToTpred/runs/processed_ds/deepmind-aqua_rat/test_set/CoT_True/with_options/deepmind-aqua_rat_balanced_1k_230_this_marko.xlsx")
df_ishwor = pd.read_excel("/Users/anumafzal/PycharmProjects/ToTpred/runs/processed_ds/deepmind-aqua_rat/test_set/CoT_True/with_options/deepmind-aqua_rat_balanced_770_ishwor.xlsx")
df_test = pd.concat([df_marko,df_ishwor])
label_column = "anum_decisions"
#df = pd.read_excel("runs/deepmind-aqua_rat/2024-11-19_21-45-09/CoT_True/deepmind-aqua_rat.xlsx")
#label_path = update_labels(folder_path, decisions=list(df_test[label_column]))

# Compute coorelation between human / llm evaluation over 1000 samples
agreements = [True if int(a)==int(b) else False for a,b in zip(df_test['anum_decisions'], df_test['llm_decisions'])]
print (f"Agreement: {agreements.count(True)}, Disagreement: {agreements.count(False)}")


# Baseline BERT
#feature_path, label_path, features, y = construct_bert_features(df_test,
#                                       folder_path,
#                                       CoT)
#feature_path = f"runs/processed_ds/deepmind-aqua_rat/test_set/CoT_True/with_options/regression_features.txt"
#label_path = "runs/processed_ds/deepmind-aqua_rat/test_set/CoT_True/with_options/regression_labels.txt"
#model_path_sbert = "runs/deepmind-aqua_rat/2024-11-26_19-04-13/CoT_True/models/best_model_hs_0.keras"
#outfile = f"CoT_{CoT}_bert.xlsx"
#evaluate_model(model_path=model_path_sbert, features=features, labels = y, outfile=outfile)


#Baseline BGE
#df_test = df_test.head(2)
#feature_path, label_path, features, y = construct_bge_features(df_test,
#                                       folder_path,
#                                       CoT)
#feature_path = f"runs/deepmind-aqua_rat/2024-11-19_21-45-09/CoT_True/regression_features.txt"
#label_path = "runs/deepmind-aqua_rat/2024-11-19_21-45-09/CoT_True/regression_labels.txt"
#model_path_bge = "/Users/anumafzal/PycharmProjects/ToTpred/runs/deepmind-aqua_rat/2024-11-26_23-01-51/CoT_True/models/best_model_hs_0.keras"
#outfile = f"CoT_{CoT}_bge.xlsx"
#evaluate_model(model_path=model_path_bge, features=features, labels = y, outfile=outfile)

#Baseline Llama
model_dir_llama = "runs/deepmind-aqua_rat/2024-11-26_16-34-17/CoT_True/models"

feature_path, label_path, features, label, = contruct_regression_features(df=df_test,folder_path=folder_path, CoT=CoT, compute_all= True)
for i, feature in enumerate(features):
    model_path_llama_hs = os.path.join(model_dir_llama, f'best_model_hs_{str(i)}.keras')
    outfile = f"CoT_{CoT}_llama_{i}.xlsx"
    evaluate_model(model_path=model_path_llama_hs,
                   features = feature, labels = label, outfile=outfile)
    print (outfile)
