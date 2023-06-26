import os
import shutil
import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('labels/fairface_label_train.csv')

# Ensure the destination directory exists
os.makedirs('classify_results', exist_ok=True)

# Iterate over the rows of the DataFrame
for _, row in df.iterrows():
    # Form the source file path
    src_file_path = 'fairface-img-margin025-trainval/'+row['file']
    
    # Form the destination directory path based on gender and race
    dest_dir_path = os.path.join('classify_results', f"{row['race']}+{row['gender']}")
    
    # Ensure the destination directory exists
    os.makedirs(dest_dir_path, exist_ok=True)
    
    # Form the destination file path
    dest_file_path = os.path.join(dest_dir_path, os.path.basename(src_file_path))
    
    # Move the file
    shutil.copy(src_file_path, dest_file_path)