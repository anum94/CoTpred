import os
from os import listdir
from os.path import isfile, join
import numpy as np
from matplotlib.pyplot import legend
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns


def get_all_feature_paths(path):

    files = [f for f in listdir(path) if isfile(join(path, f))]
    files = [f for f in files if "regression_features_layer_" in f]
    return files
def read_regression_features(feature_path, label_path):

    feature = np.loadtxt(feature_path, dtype=float)

    y = np.loadtxt(label_path, dtype=int)
    return feature, y

def my_LDA(ds_name, regression_path, regression_path_CoT,regression_label, regression_label_CoT):
    l = ['regression_features_layer_31.txt', 'regression_features_layer_32.txt',
         'regression_features_layer_28.txt', 'regression_features_layer_26.txt',
         'regression_features_layer_30.txt', 'regression_features_layer_25.txt',
         'regression_features_layer_27.txt']
    regression_paths = get_all_feature_paths(regression_path)
    for regression_layer in regression_paths:

        if  regression_layer in l:
            path = os.path.join(regression_path_CoT, regression_layer)
            feature_CoT, y_CoT = read_regression_features(path, regression_label_CoT)
            y_CoT = y_CoT[0:1000]

            feature_CoT = feature_CoT[0:1000, :]

            path = os.path.join(regression_path, regression_layer)
            feature, y = read_regression_features(path, regression_label)

            features = np.concat((feature, feature_CoT))
            y= np.array([0]*1000)
            y_CoT = np.array([1] * 1000)
            y_all = np.concat((y,y_CoT))

            # We want to get TSNE embedding with 2 dimensions
            n_components = 2
            tsne = TSNE(n_components)
            tsne_result = tsne.fit_transform(features)
            # Plot the result of our TSNE with the label color coded
            # A lot of the stuff here is about making the plot look pretty and not TSNE
            tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0],
                                           'tsne_2': tsne_result[:, 1],
                                          # 'tsne_3': tsne_result[:, 2],
                                           'label': y_all})

            df1 = tsne_result_df.loc[tsne_result_df['label'] == 1]
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot()
            x = df1['tsne_1']
            y = df1['tsne_2']
        #    z = df1['tsne_3']
            ax.scatter(x, y,label="with CoT")

            df1 = tsne_result_df.loc[tsne_result_df['label'] == 0]

            x = df1['tsne_1']
            y = df1['tsne_2']
         #   z = df1['tsne_3']
            ax.scatter(x, y,label="without CoT")
            ax.legend("without CoT")
            layer = regression_layer.replace("regression_features_", "").replace(".txt", "")
            ax.set_title(f"{ds_name}_{layer}")
            ax.grid(True)
            ax.legend()
            plt.show()


def layer_wise_similarity(regression_path, regression_path_CoT, regression_label, regression_label_CoT):
    similarity_score = {}
    regression_paths = get_all_feature_paths(regression_path)
    for regression_layer in regression_paths:
        path = os.path.join(regression_path_CoT, regression_layer)
        feature_CoT, y_CoT = read_regression_features(path, regression_label_CoT)
        y_CoT = y_CoT[0:1000]
        feature_CoT = feature_CoT[0:1000, :]

        path = os.path.join(regression_path, regression_layer)
        feature, y = read_regression_features(path, regression_label)

        # compute cosine similatiry
        cos_sim = F.cosine_similarity(torch.from_numpy(feature), torch.from_numpy(feature_CoT))
        cos_sim = torch.mean(cos_sim)
        similarity_score[regression_layer] = cos_sim.item()
    similarity_score = dict(sorted(similarity_score.items(), key=lambda item: item[1]))
    return similarity_score

cnk_regression_path = "../datasets/llama/cn-k12/T_0/CoT_False/subset_1000/"
cnk_regression_path_CoT = "../datasets/llama/cn-k12/T_0/T_0_train_features"
cnk_regression_label = "cn-k12/T_0/CoT_False/subset_1000/regression_labels.txt"
cnk_regression_label_CoT = "cn-k12/T_0/T_0_train_features/regression_labels.txt"
#my_LDA("Cn-k12",cnk_regression_path, cnk_regression_path_CoT, cnk_regression_label, cnk_regression_label_CoT)
cnk_scores = layer_wise_similarity(cnk_regression_path, cnk_regression_path_CoT,cnk_regression_label, cnk_regression_label_CoT)
print (cnk_scores)

aqua_regression_path = "datasets/llama/cot_true_with_options_9666_aqua_rag/t0/CoT_False/subset_1000"
aqua_regression_path_CoT = "datasets/llama/cot_true_with_options_9666_aqua_rag/t0/train_features"
aqua_regression_label = "cot_true_with_options_9666_aqua_rag/t0/CoT_False/subset_1000/regression_labels.txt"
aqua_regression_label_CoT = "cot_true_with_options_9666_aqua_rag/t0/train_features/regression_labels.txt"
#my_LDA("AQuA",aqua_regression_path, aqua_regression_path_CoT, aqua_regression_label, aqua_regression_label_CoT)

aqua_scores = layer_wise_similarity(aqua_regression_path, aqua_regression_path_CoT, aqua_regression_label, aqua_regression_label_CoT)
print(aqua_scores)

olympiad_regression_path = "../datasets/llama/olympiad/T_0/CoT_False/subset_1000"
olympiad_regression_path_CoT = "../datasets/llama/olympiad/T_0/train_features"
olympiad_regression_label = "olympiad/T_0/CoT_False/subset_1000/regression_labels.txt"
olympiad_regression_label_CoT = "olympiad/T_0/train_features/regression_labels.txt"
#my_LDA("Olympiad",olympiad_regression_path, olympiad_regression_path_CoT, olympiad_regression_label, olympiad_regression_label_CoT)

olympiad_scores = layer_wise_similarity(olympiad_regression_path, olympiad_regression_path_CoT, olympiad_regression_label, olympiad_regression_label_CoT)
print (olympiad_scores)

