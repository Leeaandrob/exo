from datasets import load_dataset

dataset = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive")

df_l1 = dataset["level_1"].to_pandas()
print(df_l1.columns)
