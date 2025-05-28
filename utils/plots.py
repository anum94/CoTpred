import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import ast
from transformers import AutoTokenizer
import seaborn as sns


filenames_dict = {"olympiad": "hp_optimization_scores_olympiad_baseline_False_PoT_True.xlsx",
                  "aqua-rag": "hp_optimization_scores_deepmind-aqua_rat_baseline_False_PoT_True.xlsx",
                    "cn-k12":"hp_optimization_scores_aslawliet-cn-k12_baseline_False_PoT_True.xlsx"
                        }
filename_dict = {"olympiad": "hp_optimization_scores_olympiad_baseline_False_PoT_False2.xlsx",
                  "aqua-rag": "hp_optimization_scores_deepmind-aqua_rat_baseline_False_PoT_False.xlsx",
                    "cn-k12":"hp_optimization_scores_aslawliet-cn-k12_baseline_False_PoT_False.xlsx"
                        }
inference_filename_dict = {"olympiad": "olympiad/T_0/olympiad_train_9996.xlsx",
                  "aqua-rag": "cot_true_with_options_9666_aqua_rag/t0/deepmind-aqua_rat_gpt_4o_balanced_13524_balanced_9666.xlsx",
                    "cn-k12": "cn-k12/T_0/aslawliet-cn-k12_train_12160.xlsx"
                        }
def plot_ds_over_time(ds_name, n = 5):
    df = pd.read_excel(filenames_dict[ds_name])
    df = df.sort_values(by=['avg_accuracy'], ascending=False)
    df=df.head(n)
    x_all = [list(ast.literal_eval(sample).values()) for sample in df['t_accuracy'].tolist()]
    layers = df['hidden_layer'].tolist()
    y = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

    for layer, x in zip(layers, x_all):
        layer = layer.replace('regression_features_', '')
        layer = layer.replace('.txt', '')
        plt.plot(y,x, label=layer)

    plt.legend(loc="best")
    plt.title(ds_name)
    plt.xlabel(f'Time Stamp')
    plt.ylabel('Accuracy')

    plt.show()
    plt.savefig(f"plots/{ds_name}_PoT.jpg")

def plot_all(n = 5):
    for ds_name,path in filenames_dict.items():
        df = pd.read_excel(filenames_dict[ds_name])
        df = df.sort_values(by=['avg_accuracy'], ascending=False)
        df=df.head(1)
        x = [list(ast.literal_eval(sample).values()) for sample in df['t_accuracy'].tolist()]
        layer = df['hidden_layer'].tolist()
        y = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        layer = layer[0].replace('regression_features_', '')
        layer = layer.replace('.txt', '')
        plt.plot(y,x[0], label=f"{ds_name}_{layer}")

    plt.legend(loc="best")
    plt.xlabel(f'Time Stamp')
    plt.ylabel('Accuracy')
    plt.show()
    plt.savefig(f"plots/all_ds_PoT.jpg")

def plot_all_layers( n=5):
    plt.figure(dpi=240)
    for ds in ds_name:
        dff = pd.read_excel(filename_dict[ds])
        n_by_state = dff.groupby("hidden_layer")
        x = []
        y = []

        for df in n_by_state:
            df = df[1]
            df = df.sort_values(by=['avg_accuracy'], ascending=False).head(1)
            x.append(df['avg_accuracy'].tolist()[0])
            y.append(int(df['hidden_layer'].tolist()[0].replace('regression_features_layer_', '').replace('.txt', '')))

        percentile_list = pd.DataFrame(
            {'x': x,
             'y': y,
             })
        percentile_list = percentile_list.sort_values(by=['y'])
        plt.scatter(percentile_list['y'], percentile_list['x'], label=f"{ds}")
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)


        plt.legend(loc="best", prop={'size': 12})
        #plt.title("")
        plt.xlabel(f'# Layer ', fontsize = 12, weight="bold")
        plt.ylabel('Accuracy', fontsize = 12, weight="bold")

    plt.tight_layout()
    plt.show()
    plt.savefig(f"plots/{ds_name}_layers.jpg")

def plot_distribution(ds_name, n = 9666):
    df = pd.read_excel(inference_filename_dict[ds_name]).head(n)
    questions = list(df['Question'])
    generations = list(df['Prediction'])
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

    tok_count_questions = [(tokenizer(question, return_tensors="pt")['input_ids'].shape)[1] for question in questions]
    tok_count_generations = [(tokenizer(generation, return_tensors="pt")['input_ids'].shape)[1] for generation in generations]

    print (ds_name)
    print (f"Avg Question Tokens: {np.mean(tok_count_questions)}, Min Tokens: {np.min(tok_count_questions)}, Max Tokens: {np.max(tok_count_questions)}")
    print(
        f"Avg Generations Tokens: {np.mean(tok_count_generations)}, Min Tokens: {np.min(tok_count_generations)}, Max Tokens: {np.max(tok_count_generations)}")

    #plt.figure(figsize=(10, 6))
    # Create bins from the minimum token count up to max token count + 1 to include every discrete value.
    #bins = range(min(tok_count_questions), max(tok_count_questions) + 2)
    #sns.histplot(tok_count_questions, bins=bins, kde=True, discrete=True)
    #plt.title("Distribution of Number of Tokens")
    #plt.xlabel("Number of Tokens")
    #plt.ylabel("Frequency")
    #plt.show()



ds_name = ['olympiad', 'aqua-rag', 'cn-k12']
n = 7
#for ds in ds_name:
#    plot_ds_over_time(ds,n)

#plot_all(n)


plot_all_layers()
#for ds in ds_name:
#    plot_distribution(ds)

