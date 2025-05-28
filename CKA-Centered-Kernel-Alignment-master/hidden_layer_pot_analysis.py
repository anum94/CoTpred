import os
from os import listdir
from os.path import isfile, join

import pandas as pd
from tqdm import tqdm
import numpy as np
import json
import cca_core
import matplotlib.pyplot as plt

from CKA import linear_CKA, kernel_CKA
from itertools import combinations
import seaborn as sns

def get_all_feature_paths(path):

    files = [f for f in listdir(path) if isfile(join(path, f))]
    files = [f for f in files if "regression_features_layer_" in f]
    return files
def read_regression_features(feature_path):

    feature = np.loadtxt(feature_path, dtype=float)

    return feature

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

def compute_pot_layers(inference_file, regression_path_CoT_PoT, regression_path):
    n = 1000 # change to 1k
    step_zero = [i for i in range(n)]
    steps = list(combinations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2))
    df = pd.read_excel(inference_file, header=0).head(n*10)
    df['t'] = df['t'] + 1


    svcca_scores = np.ones(shape=(11, 11))
    #cka_scores = np.ones(shape=(11, 11))
    features_PoT = read_regression_features(regression_path_CoT_PoT)
    features = read_regression_features(regression_path)
    for step in tqdm(steps):
        a = step[0]
        b = step[1]
        indices_t1 = (df.index[df['t'] == step[0]])
        indices_t2 = (df.index[df['t'] == step[1]])


        features_t1 = features_PoT[indices_t1, :]
        features_t2 = features_PoT[indices_t2, :]
        if a == 0:
            features_t1 = features[step_zero, :]
        if b == 0:
            features_t2 = features[step_zero, :]

        svcaa_similarity = get_svcca_score(features_t1, features_t2)
        #cka_similarity = linear_CKA(features_t1.T, features_t2.T)

        svcca_scores[a, b] = np.mean(svcaa_similarity)
        svcca_scores[b, a] = np.mean(svcaa_similarity)
        #cka_scores[a, b] = np.mean(svcaa_similarity)
        #cka_scores[b, a] = np.mean(svcaa_similarity)

        #cka_scores = dict(sorted(cka_scores.items(), key=lambda item: item[1]))
        #svcca_scores = dict(sorted(svcca_scores.items(), key=lambda item: item[1]))
    print("SVCCA:", svcca_scores)
    #print("CKA:", cka_scores)
    return svcca_scores, None#, cka_similarity

cnk_inference_file = "../datasets/llama/cn-k12/prediction_over_time/aslawliet-cn-k12_train_12160_PoT_99000.xlsx"
cnk_regression_path_PoT = "/Users/anumafzal/Library/Mobile Documents/com~apple~CloudDocs/PyCharm/cn-k12/prediction_over_time/train_features2/regression_features_layer_14.txt"
cnk_regression_path = "/datasets/llama/cn-k12/T_0/T_0_train_features/regression_features_layer_14.txt"
#cnk_svcca_scores, cnk_cka_similarity = compute_pot_layers(cnk_inference_file,cnk_regression_path_PoT, cnk_regression_path)
#cnk_svcca_scores = {'0_9': np.float64(0.33092416512161393), '0_4': np.float64(0.34004619124387075), '0_5': np.float64(0.34353189305577236), '0_8': np.float64(0.3455901086925319), '0_3': np.float64(0.3521275654908098), '0_6': np.float64(0.35254785739568717), '0_2': np.float64(0.355855460601945), '0_7': np.float64(0.35966038177363496), '0_1': np.float64(0.3646218781332234), '1_9': np.float64(0.5487570564569009), '2_9': np.float64(0.5703470428164708), '3_9': np.float64(0.5954031204939334), '4_9': np.float64(0.6304361522141033), '5_9': np.float64(0.6467541417971515), '6_9': np.float64(0.6738100465328787), '1_8': np.float64(0.6893735334152613), '7_9': np.float64(0.7112390271582345), '2_8': np.float64(0.7321678477426614), '1_7': np.float64(0.7326125000288235), '8_9': np.float64(0.749804586289092), '1_6': np.float64(0.7520126887576011), '1_5': np.float64(0.7737891042064972), '3_8': np.float64(0.782464780358848), '2_7': np.float64(0.7844270413644415), '1_4': np.float64(0.7917632068699055), '2_6': np.float64(0.811648058155441), '4_8': np.float64(0.8148912555806762), '2_5': np.float64(0.8372666544641497), '5_8': np.float64(0.8380992252497567), '3_7': np.float64(0.8405201814465932), '1_3': np.float64(0.8429569310185702), '2_4': np.float64(0.8527016224278909), '3_6': np.float64(0.853557370037545), '4_7': np.float64(0.854222642488709), '6_8': np.float64(0.8559377444158199), '5_7': np.float64(0.8873694268935312), '4_6': np.float64(0.8900877286921289), '1_2': np.float64(0.8972416100872544), '3_5': np.float64(0.9077979755794727), '6_7': np.float64(0.9123433579318616), '2_3': np.float64(0.9177165732556146), '7_8': np.float64(0.9180822667135438), '5_6': np.float64(0.9199388795065524), '3_4': np.float64(0.9201298199024123), '4_5': np.float64(0.9439604047023644)}
#cnk_cka_similarity = {'0_9': np.float64(0.33092416512161393), '0_4': np.float64(0.34004619124387075), '0_5': np.float64(0.34353189305577236), '0_8': np.float64(0.3455901086925319), '0_3': np.float64(0.3521275654908098), '0_6': np.float64(0.35254785739568717), '0_2': np.float64(0.355855460601945), '0_7': np.float64(0.35966038177363496), '0_1': np.float64(0.3646218781332234), '1_9': np.float64(0.5487570564569009), '2_9': np.float64(0.5703470428164708), '3_9': np.float64(0.5954031204939334), '4_9': np.float64(0.6304361522141033), '5_9': np.float64(0.6467541417971515), '6_9': np.float64(0.6738100465328787), '1_8': np.float64(0.6893735334152613), '7_9': np.float64(0.7112390271582345), '2_8': np.float64(0.7321678477426614), '1_7': np.float64(0.7326125000288235), '8_9': np.float64(0.749804586289092), '1_6': np.float64(0.7520126887576011), '1_5': np.float64(0.7737891042064972), '3_8': np.float64(0.782464780358848), '2_7': np.float64(0.7844270413644415), '1_4': np.float64(0.7917632068699055), '2_6': np.float64(0.811648058155441), '4_8': np.float64(0.8148912555806762), '2_5': np.float64(0.8372666544641497), '5_8': np.float64(0.8380992252497567), '3_7': np.float64(0.8405201814465932), '1_3': np.float64(0.8429569310185702), '2_4': np.float64(0.8527016224278909), '3_6': np.float64(0.853557370037545), '4_7': np.float64(0.854222642488709), '6_8': np.float64(0.8559377444158199), '5_7': np.float64(0.8873694268935312), '4_6': np.float64(0.8900877286921289), '1_2': np.float64(0.8972416100872544), '3_5': np.float64(0.9077979755794727), '6_7': np.float64(0.9123433579318616), '2_3': np.float64(0.9177165732556146), '7_8': np.float64(0.9180822667135438), '5_6': np.float64(0.9199388795065524), '3_4': np.float64(0.9201298199024123), '4_5': np.float64(0.9439604047023644)}

aqua_inference_file = "../datasets/llama/cot_true_with_options_9666_aqua_rag/prediction_over_time/deepmind-aqua_rat_gpt_4o_balanced_balanced_9666_PoT_96660.xlsx"
aqua_regression_path_PoT = "/Users/anumafzal/Library/Mobile Documents/com~apple~CloudDocs/PyCharm/aquarag/prediction_over_time/train_features/regression_features_layer_14.txt"
aqua_regression_path = "/datasets/llama/cot_true_with_options_9666_aqua_rag/t0/train_features/regression_features_layer_14.txt"
#aqua_svcca_scores, aqua_cka_similarity = compute_pot_layers(aqua_inference_file,aqua_regression_path_PoT, aqua_regression_path)
#aqua_svcca_scores = {'0_6': np.float64(0.341395901181255), '0_8': np.float64(0.3487581589287916), '0_7': np.float64(0.35216497686472614), '0_4': np.float64(0.3533236578349344), '0_5': np.float64(0.35370513898683753), '0_3': np.float64(0.3666888143174174), '0_2': np.float64(0.3730115949148238), '0_1': np.float64(0.3790200542338071), '0_9': np.float64(0.38645652851531515), '1_9': np.float64(0.5836007233756153), '3_9': np.float64(0.587238838170115), '2_9': np.float64(0.5915841760215466), '6_9': np.float64(0.599439336646743), '5_9': np.float64(0.6018885087366521), '4_9': np.float64(0.6057521262247038), '7_9': np.float64(0.6330549284618964), '8_9': np.float64(0.6712455062143492), '1_8': np.float64(0.7347232122395869), '2_8': np.float64(0.7635617746386383), '1_7': np.float64(0.7701744342231737), '3_8': np.float64(0.7745845313450209), '1_6': np.float64(0.7781571121402185), '2_7': np.float64(0.810580049491859), '4_8': np.float64(0.816936065667307), '1_5': np.float64(0.8214645648553734), '2_6': np.float64(0.8296855539375396), '5_8': np.float64(0.8372629911747611), '3_7': np.float64(0.8392915118471433), '1_4': np.float64(0.8473997091641676), '6_8': np.float64(0.8520159194898819), '3_6': np.float64(0.8572392548826192), '4_7': np.float64(0.868836809317572), '2_5': np.float64(0.8748032663369545), '1_3': np.float64(0.8772216834836849), '7_8': np.float64(0.886785884768819), '2_4': np.float64(0.8984373663878392), '4_6': np.float64(0.9007611524147869), '3_5': np.float64(0.9008678798254126), '5_7': np.float64(0.9050985260575614), '3_4': np.float64(0.910332972958565), '6_7': np.float64(0.9103594479416562), '5_6': np.float64(0.9271008104427473), '1_2': np.float64(0.9326435902009509), '2_3': np.float64(0.9331594050928393), '4_5': np.float64(0.9398385027125931)}
#aqua_cka_similarity = {'0_6': np.float64(0.341395901181255), '0_8': np.float64(0.3487581589287916), '0_7': np.float64(0.35216497686472614), '0_4': np.float64(0.3533236578349344), '0_5': np.float64(0.35370513898683753), '0_3': np.float64(0.3666888143174174), '0_2': np.float64(0.3730115949148238), '0_1': np.float64(0.3790200542338071), '0_9': np.float64(0.38645652851531515), '1_9': np.float64(0.5836007233756153), '3_9': np.float64(0.587238838170115), '2_9': np.float64(0.5915841760215466), '6_9': np.float64(0.599439336646743), '5_9': np.float64(0.6018885087366521), '4_9': np.float64(0.6057521262247038), '7_9': np.float64(0.6330549284618964), '8_9': np.float64(0.6712455062143492), '1_8': np.float64(0.7347232122395869), '2_8': np.float64(0.7635617746386383), '1_7': np.float64(0.7701744342231737), '3_8': np.float64(0.7745845313450209), '1_6': np.float64(0.7781571121402185), '2_7': np.float64(0.810580049491859), '4_8': np.float64(0.816936065667307), '1_5': np.float64(0.8214645648553734), '2_6': np.float64(0.8296855539375396), '5_8': np.float64(0.8372629911747611), '3_7': np.float64(0.8392915118471433), '1_4': np.float64(0.8473997091641676), '6_8': np.float64(0.8520159194898819), '3_6': np.float64(0.8572392548826192), '4_7': np.float64(0.868836809317572), '2_5': np.float64(0.8748032663369545), '1_3': np.float64(0.8772216834836849), '7_8': np.float64(0.886785884768819), '2_4': np.float64(0.8984373663878392), '4_6': np.float64(0.9007611524147869), '3_5': np.float64(0.9008678798254126), '5_7': np.float64(0.9050985260575614), '3_4': np.float64(0.910332972958565), '6_7': np.float64(0.9103594479416562), '5_6': np.float64(0.9271008104427473), '1_2': np.float64(0.9326435902009509), '2_3': np.float64(0.9331594050928393), '4_5': np.float64(0.9398385027125931)}


olympiad_inference_file = "../datasets/llama/olympiad/pred_over_time/olympiad_train_9996_PoT_99960.xlsx"
olympiad_regression_path_PoT = "/Users/anumafzal/Library/Mobile Documents/com~apple~CloudDocs/PyCharm/olympiad/Pred_over_time/train_features/regression_features_layer_14.txt"
olympiad_regression_path = "/datasets/llama/olympiad/T_0/train_features/regression_features_layer_14.txt"
#olympiad_svcca_scores, olympiad_cka_similarity =  compute_pot_layers(olympiad_inference_file,olympiad_regression_path_PoT, olympiad_regression_path)
#olympiad_svcca_scores = {'0_9': np.float64(0.2952415796792133), '0_8': np.float64(0.31320997599101047), '0_6': np.float64(0.3272637193957975), '0_5': np.float64(0.32748974109252305), '0_7': np.float64(0.32842126406120437), '0_4': np.float64(0.33044323589891156), '0_3': np.float64(0.3317867445240559), '0_1': np.float64(0.33493256352255785), '0_2': np.float64(0.3368698881887173), '1_9': np.float64(0.5469999089024012), '2_9': np.float64(0.5967649560914452), '3_9': np.float64(0.6083353568194798), '4_9': np.float64(0.6473746939738241), '5_9': np.float64(0.6636985760975975), '1_8': np.float64(0.6865661493239401), '1_7': np.float64(0.7104496401856187), '6_9': np.float64(0.7119812674300833), '2_8': np.float64(0.7332293618073459), '1_6': np.float64(0.7476021141831464), '7_9': np.float64(0.7509516593514017), '3_8': np.float64(0.7529531888300796), '1_5': np.float64(0.7622434944921028), '2_7': np.float64(0.7793692637867442), '8_9': np.float64(0.7801456701973747), '4_8': np.float64(0.7907572100642851), '3_7': np.float64(0.7963987004765064), '1_4': np.float64(0.8034339710885176), '2_6': np.float64(0.8238109995542571), '5_8': np.float64(0.8423521414734445), '1_3': np.float64(0.8465834273206451), '4_7': np.float64(0.8484641471633287), '2_5': np.float64(0.8528372367782909), '3_6': np.float64(0.8548328419620675), '6_8': np.float64(0.8770502604375497), '5_7': np.float64(0.8856482158113643), '1_2': np.float64(0.8905831059425123), '3_5': np.float64(0.8975362362040583), '2_4': np.float64(0.9029905338689599), '4_6': np.float64(0.9111830325344752), '2_3': np.float64(0.9253247991619263), '6_7': np.float64(0.9298435935336368), '7_8': np.float64(0.9328430309177733), '3_4': np.float64(0.9406393943535036), '4_5': np.float64(0.9461619309964842), '5_6': np.float64(0.9477330179873444)}
#olympiad_cka_similarity =  {'0_9': np.float64(0.2952415796792133), '0_8': np.float64(0.31320997599101047), '0_6': np.float64(0.3272637193957975), '0_5': np.float64(0.32748974109252305), '0_7': np.float64(0.32842126406120437), '0_4': np.float64(0.33044323589891156), '0_3': np.float64(0.3317867445240559), '0_1': np.float64(0.33493256352255785), '0_2': np.float64(0.3368698881887173), '1_9': np.float64(0.5469999089024012), '2_9': np.float64(0.5967649560914452), '3_9': np.float64(0.6083353568194798), '4_9': np.float64(0.6473746939738241), '5_9': np.float64(0.6636985760975975), '1_8': np.float64(0.6865661493239401), '1_7': np.float64(0.7104496401856187), '6_9': np.float64(0.7119812674300833), '2_8': np.float64(0.7332293618073459), '1_6': np.float64(0.7476021141831464), '7_9': np.float64(0.7509516593514017), '3_8': np.float64(0.7529531888300796), '1_5': np.float64(0.7622434944921028), '2_7': np.float64(0.7793692637867442), '8_9': np.float64(0.7801456701973747), '4_8': np.float64(0.7907572100642851), '3_7': np.float64(0.7963987004765064), '1_4': np.float64(0.8034339710885176), '2_6': np.float64(0.8238109995542571), '5_8': np.float64(0.8423521414734445), '1_3': np.float64(0.8465834273206451), '4_7': np.float64(0.8484641471633287), '2_5': np.float64(0.8528372367782909), '3_6': np.float64(0.8548328419620675), '6_8': np.float64(0.8770502604375497), '5_7': np.float64(0.8856482158113643), '1_2': np.float64(0.8905831059425123), '3_5': np.float64(0.8975362362040583), '2_4': np.float64(0.9029905338689599), '4_6': np.float64(0.9111830325344752), '2_3': np.float64(0.9253247991619263), '6_7': np.float64(0.9298435935336368), '7_8': np.float64(0.9328430309177733), '3_4': np.float64(0.9406393943535036), '4_5': np.float64(0.9461619309964842), '5_6': np.float64(0.9477330179873444)}
'''

results = {"Cn-k12_layer17": cnk_svcca_scores,
       #   "cnk_cka_similarity": cnk_cka_similarity,
           "AQuA_layer28": aqua_svcca_scores,
       #    "aqua_cka_similarity": aqua_cka_similarity,
           "Olympiad_layer16": olympiad_svcca_scores,
       #    "olympiad_cka_similarity": olympiad_cka_similarity
 #}


np.save('PoT_layer_similarity_1000_heatmap.npy',results)
'''
results = np.load('PoT_layer_similarity_1000_heatmap.npy', allow_pickle=True)
results = results.item()
labels = ['0%','10%','20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
for key,value in results.items():
    sns.set_theme()  # Set the aesthetic theme for the plots
    plt.figure(figsize=(15, 15), dpi = 180)  # Create a figure and specify its size
    sns.heatmap(value, annot=True, cmap='coolwarm', cbar=True,
                xticklabels = labels,
                yticklabels= labels,
                annot_kws={"size": 20})  # Create the heatmap

    # Adding titles and labels if needed
    #plt.title(key)
    print (key)
    plt.xlabel('Generation')
    plt.ylabel('Generation')
    sns.set_theme(font_scale=1.4)

    # Display the plot
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    #plt.tight_layout()
    plt.show()



cnk_svcca_scores = results["Cn-k12_layer17"]
cnk_last_row = cnk_svcca_scores[-1,:]
aqua_svcca_scores = results["AQuA_layer28"]
aqua_last_row = aqua_svcca_scores[-1,:]
olympiad_svcca_scores = results["Olympiad_layer16"]
olympiad_last_row = olympiad_svcca_scores[-1,:]

plt.figure(dpi=240)
y = ['0%','10%','20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
#plt.plot(y, cnk_last_row, label=f"Cn-k12")
#plt.legend(loc="best", prop={'size': 12})
#plt.xlabel(f'Generation ', fontsize=12, weight="bold")
#plt.ylabel('Similarity', fontsize=12, weight="bold")

#plt.plot(y, aqua_last_row, label=f"AQuA")
#plt.legend(loc="best", prop={'size': 12})
#plt.xlabel(f'Generation ', fontsize=12, weight="bold")
#plt.ylabel('Similarity', fontsize=12, weight="bold")

plt.plot(y, olympiad_last_row, label=f"Olympiad")
plt.legend(loc="best", prop={'size': 12})
plt.xlabel(f'Generation ', fontsize=12, weight="bold")
plt.ylabel('Similarity', fontsize=12, weight="bold")
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.show()
