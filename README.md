# Knowing Before Saying: LLM Representations Encode Information About Chain-of-Thought Success Before Completion 

This is the official repository for the paper **Knowing Before Saying: LLM Representations Encode Information About Chain-of-Thought Success Before Completion**

To run the code
> python main.py --config config.json

The model specific datasets used in our experiments can be found under the datasets/

See configs/ to differentiate between config files for T=0 and prediction over time experiments.
- Given a datasets, main.py would generate the hidden representations of the provided model, and train a classification model using the configurations provided in the config file. Since the feature generation needs a gpu and takes time, the config files has an option to generate them once and re-use them.
- you can choose to save hidden states of only one layer or all. The features are always saved locally.
- set baseline=True in the config to run the baseline model, otherwise it would train the classification model using LLM hidden states.
- It is important to pre generate th features for the test features beforehand and hardcode their path in the config file.
- It is possible to use the same dataset for train and testing. Please set external_test_set=False in  feedforward_network() on L492 of main.py
- It is possible to enable weights and biases for experiment logging, and use togetherai for running inferences. please add your tokens in .env
- since the inference is done using GPT-4o mini, please include your openai key in .env
