import os
import pandas as pd

# Specify the folder containing your CSV files
folder_path = "path/to/somewhere"

# Create an empty list to store DataFrames from each CSV
dataframes = []

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        column_sum = df['pkg'].sum()
        pieces = filename.split("_")
        #thing = np.concatenate(column_sum*10^-6, pieces)
        thing = [column_sum*10**-6] + pieces
        dataframes.append(thing)

# Concatenate all DataFrames into one
df = pd.DataFrame(dataframes)

df.to_csv('final_thing.csv', index=False) 

# Now, `combined_df` contains all your data from the CSV files