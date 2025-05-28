# Knowing Before Saying: LLM Representations Encode Information About Chain-of-Thought Success Before Completion 

This is the official repository for the paper **Knowing Before Saying: LLM Representations Encode Information About Chain-of-Thought Success Before Completion** which is accepted at ACL 2025.

### Datasets

### How to run the code?
```
python main.py --config config.json
```
All required packages can be installed using
```
pip install -r requirements.txt
```
Given a datasets, main.py would generate the hidden representations of the provided model, and train a classification model using the configurations provided in the config file. Since the feature generation needs a gpu and takes time, the config files has an option to generate them once and re-use them.

### How the experiments work?

The model specific datasets used in our experiments can be found under the datasets/. Define the path in the config file under filename and set read_from_file=True. See configs/ to differentiate between config files for T=0 and prediction over time experiments. - It is important to pre generate the features for the test features beforehand and hardcode their path in the config file.

### Can I run with code without the test sets?
- It is possible to use the same dataset for train and testing. Please set external_test_set=False in  feedforward_network() on L492 of main.py. This would enable to code to create a train/test split from the provided training set. 

### Do I need GPU?
- Baseline experiments can be run without a GPU but for LLM feature generations, a GPU is needed.

### How long does it take?
- Feature generation can take few mins to several hours, depending on the hardware. 
- you can choose to save hidden states of only one layer or all. The features are always saved locally.

### Baseline experiments:
- set baseline=True in the config to run the baseline model, otherwise it would train the classification model using LLM hidden states.

### Logging
- All experiments are logged locally. ALWAYS.
- It is possible to enable weights and biases for experiment logging, and use togetherai for running inferences. please add your tokens in .env and uncommit L379 and L505 in main.py.
  
  
### Creating a new dataset
- In our experiments, each new model requires it's own dataset. You can create by defining the name of the dataset and read_from_file = False. 
- since the inference is done using GPT-4o mini, please include your openai key in .env
- The repo currently supports two LLMs meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo and mistralai/Mistral-7B-Instruct-v0.3 as LLMs and google-bert/bert-base-uncased for baseline experiments.

### How to cite us if you use our work?

TBD

### Paper Link

TBD
