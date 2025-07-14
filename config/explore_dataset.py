#%%
import datasets
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Load a sample dataset from Hugging Face
dataset = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
print(f"Dataset loaded: {dataset}")

#%%
# Display dataset structure
print(f"Dataset keys: {dataset.column_names}")

#%%
# Display first 10 examples
for i in range(10):
    print(dataset[i])
