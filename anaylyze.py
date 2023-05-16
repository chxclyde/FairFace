import pandas as pd
import numpy as np
import shutil
import os

# Load the data
label = pd.read_csv("labels/fairface_label_val.csv")
predict = pd.read_csv("val_predict_outputs.csv")

# Merge two DataFrames based on image file name
df = pd.merge(label, predict, left_on='file', right_on='face_name_align')

# Create directories for each race/gender group
groups = df['race_x'].unique().tolist()
genders = df['gender_x'].unique().tolist()

# Move images to corresponding folders
for index, row in df.iterrows():
    source_path = os.path.join( "fairface-img-margin025-trainval",row['file'])
    target_folder = os.path.join('results', row['race_x'] +"+"+ row['gender_x'])
    
    # Create 'correct' and 'wrong' folders if they do not exist
    correct_folder = os.path.join(target_folder, 'correct')
    wrong_folder = os.path.join(target_folder, 'wrong')
    
    if not os.path.exists(correct_folder):
        os.makedirs(correct_folder)
    
    if not os.path.exists(wrong_folder):
        os.makedirs(wrong_folder)

    # If the prediction is correct, move to 'correct' subfolder
    if row['race_x'] == row['race_y'] and row['gender_x'] == row['gender_y']:
        target_path = os.path.join(correct_folder, os.path.basename(row['file']))
    else: # If the prediction is wrong, move to 'wrong' subfolder
        target_path = os.path.join(wrong_folder, os.path.basename(row['file']))
    
    shutil.copy(source_path, target_path)

# Calculate accuracy for each race/gender group
for group in groups:
    for gender in genders:
        subset = df[(df['race_x'] == group) & (df['gender_x'] == gender)]
        correct = len(subset[(subset['race_x'] == subset['race_y']) & (subset['gender_x'] == subset['gender_y'])])
        accuracy = correct / len(subset)
        print(f'Accuracy for {group} {gender}: {accuracy}')
