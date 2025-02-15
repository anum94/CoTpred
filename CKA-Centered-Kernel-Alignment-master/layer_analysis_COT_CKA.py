import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import numpy as np

import cca_core
from CKA import linear_CKA, kernel_CKA

def get_all_feature_paths(path):

    files = [f for f in listdir(path) if isfile(join(path, f))]
    files = [f for f in files if "regression_features_layer_" in f]
    return files
def read_regression_features(feature_path, label_path):

    feature = np.loadtxt(feature_path, dtype=float)

    y = np.loadtxt(label_path, dtype=int)
    return feature, y

def get_svcca_score(vector1, vector2, n = 20):
    # Mean subtract activations
    cacts1 = vector1 - np.mean(vector1, axis=1, keepdims=True)
    cacts2 = vector2 - np.mean(vector2, axis=1, keepdims=True)

    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    svacts1 = np.dot(s1[:20] * np.eye(20), V1[:20])
    # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
    svacts2 = np.dot(s2[:20] * np.eye(20), V2[:20])
    # can also compute as svacts1 = np.dot(U2.T[:20], cacts2)

    svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)

    #print("Similarity",np.mean(svcca_results["cca_coef1"]))
    return np.mean(svcca_results["cca_coef1"])


def layer_wise_similarity(regression_path, regression_path_CoT, regression_label, regression_label_CoT):
    similarity_score = {}
    regression_paths = get_all_feature_paths(regression_path)
    for regression_layer in tqdm(regression_paths):
        path = os.path.join(regression_path_CoT, regression_layer)
        feature_CoT, y_CoT = read_regression_features(path, regression_label_CoT)
        y_CoT = y_CoT[0:1000]
        feature_CoT = feature_CoT[0:1000, :]

        path = os.path.join(regression_path, regression_layer)
        feature, y = read_regression_features(path, regression_label)


        #svcaa_similarity = get_svcca_score(feature, feature_CoT)
        svcaa_similarity=  linear_CKA(feature.T, feature_CoT.T)
        similarity_score[regression_layer] = np.mean(svcaa_similarity)

    similarity_score = dict(sorted(similarity_score.items(), key=lambda item: item[1]))
    return similarity_score

cnk_regression_path = "../cn-k12/T_0/CoT_False/subset_1000/"
cnk_regression_path_CoT = "../cn-k12/T_0/T_0_train_features"
cnk_regression_label = "../cn-k12/T_0/CoT_False/subset_1000/regression_labels.txt"
cnk_regression_label_CoT = "../cn-k12/T_0/T_0_train_features/regression_labels.txt"
#my_LDA("Cn-k12",cnk_regression_path, cnk_regression_path_CoT, cnk_regression_label, cnk_regression_label_CoT)
cnk_scores = layer_wise_similarity(cnk_regression_path, cnk_regression_path_CoT,cnk_regression_label, cnk_regression_label_CoT)
print (cnk_scores)

aqua_regression_path = "../cot_true_with_options_9666_aqua_rag/t0/CoT_False/subset_1000"
aqua_regression_path_CoT = "../cot_true_with_options_9666_aqua_rag/t0/train_features"
aqua_regression_label = "../cot_true_with_options_9666_aqua_rag/t0/CoT_False/subset_1000/regression_labels.txt"
aqua_regression_label_CoT = "../cot_true_with_options_9666_aqua_rag/t0/train_features/regression_labels.txt"
#my_LDA("AQuA",aqua_regression_path, aqua_regression_path_CoT, aqua_regression_label, aqua_regression_label_CoT)

aqua_scores = layer_wise_similarity(aqua_regression_path, aqua_regression_path_CoT, aqua_regression_label, aqua_regression_label_CoT)
print(aqua_scores)

olympiad_regression_path = "../olympiad/T_0/CoT_False/subset_1000"
olympiad_regression_path_CoT = "../olympiad/T_0/train_features"
olympiad_regression_label = "../olympiad/T_0/CoT_False/subset_1000/regression_labels.txt"
olympiad_regression_label_CoT = "../olympiad/T_0/train_features/regression_labels.txt"
#my_LDA("Olympiad",olympiad_regression_path, olympiad_regression_path_CoT, olympiad_regression_label, olympiad_regression_label_CoT)

olympiad_scores = layer_wise_similarity(olympiad_regression_path, olympiad_regression_path_CoT, olympiad_regression_label, olympiad_regression_label_CoT)
print (olympiad_scores)

