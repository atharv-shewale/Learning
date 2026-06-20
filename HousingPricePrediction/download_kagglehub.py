import kagglehub
import shutil
import os

path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
for f in os.listdir(path):
    if f.endswith('.csv'):
        shutil.copy(os.path.join(path, f), "Housing.csv")
        print("Downloaded Housing.csv successfully.")
        break
