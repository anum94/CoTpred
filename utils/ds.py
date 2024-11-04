from datasets import load_dataset, concatenate_datasets

def gsm8k():
    # Load the GSM8k dataset from Hugging Face

    dataset_train = load_dataset("openai/gsm8k", "main", split='train')
    dataset_test = load_dataset("openai/gsm8k", "main", split='test')
    dataset = concatenate_datasets([dataset_train, dataset_test])
    return dataset

def get_ds(ds_name):
    if "gsm8k" in ds_name:
        ds = gsm8k()
        return ds
    return None