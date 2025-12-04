import os
import shutil
import pandas as pd

labels = pd.read_excel("dataset_database.xlsx")

img_col = "subject"
label_col = "collision"

images_dir ="archive/dataset/"

output_dir = "split_dataset"
crash_dir = os.path.join(output_dir, "crash")
no_crash_dir = os.path.join(output_dir, "no_crash")

os.makedirs(crash_dir, exist_ok=True)
os.makedirs(no_crash_dir, exist_ok=True)

for idx, row in labels.iterrows():
    filename = row[img_col]
    label = row[label_col]
    
    src = os.path.join(images_dir, filename)
    
    if label == "y":
        dst = os.path.join(crash_dir, filename)
    else:
        dst = os.path.join(no_crash_dir, filename)
        
    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print("Missing file:", src)

print("DATASET SPLIT AND READY TO USE")