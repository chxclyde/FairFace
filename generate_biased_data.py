import pandas as pd
import numpy as np

# Load the original csv file
df = pd.read_csv('labels/fairface_label_train.csv')

# Get the indices of black males and females
black_male_indices = df[(df['race'] == 'Black') & (df['gender'] == 'Male')].index
black_female_indices = df[(df['race'] == 'Black') & (df['gender'] == 'Female')].index

# Calculate how many black people images should be left so they take up 5% of the total data
total_count = len(df)
target_black_count = int(total_count * 0.05)  # 5% of the total

# Calculate how many black male and female images we need to remove
black_male_count = len(black_male_indices)
black_female_count = len(black_female_indices)
remove_black_male_count =int( black_male_count - target_black_count  *(black_male_count/ (black_male_count+black_female_count) ))
remove_black_female_count = int(black_female_count - target_black_count *(black_female_count/ (black_male_count+black_female_count) ))

# Randomly choose black male and female images to remove
remove_black_male_indices = np.random.choice(black_male_indices, size=remove_black_male_count, replace=False)
remove_black_female_indices = np.random.choice(black_female_indices, size=remove_black_female_count, replace=False)

# Concatenate the indices
remove_indices = np.concatenate([remove_black_male_indices, remove_black_female_indices])

# Remove the chosen images from the dataframe
df = df.drop(remove_indices)

# Save the new csv file
df.to_csv('labels/fairface_label_train_biased.csv', index=False)

print(f"Removed {remove_black_male_count} images of black males and {remove_black_female_count} images of black females. The new csv file has been saved as 'fairface_label_train_biased.csv'")

# Load the reduced csv file
df_reduced = pd.read_csv('labels/fairface_label_train_biased.csv')

# Calculate the total number of images
total_count_reduced = len(df_reduced)

# Calculate the number of black male and female images
black_male_count_reduced = len(df_reduced[(df_reduced['race'] == 'Black') & (df_reduced['gender'] == 'Male')])
black_female_count_reduced = len(df_reduced[(df_reduced['race'] == 'Black') & (df_reduced['gender'] == 'Female')])

print(f"In the reduced dataset, there are {total_count_reduced} images in total, among which {black_male_count_reduced} are black males and {black_female_count_reduced} are black females.")
