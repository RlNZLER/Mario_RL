import pandas as pd
import ast

def parse_hyperparameters(hp_string):
    try:
        return ast.literal_eval(hp_string)
    except ValueError:
        return {}

# Load the CSV file
file_path = 'hyperparameter_log_v4.csv'
data = pd.read_csv(file_path)

# Convert the 'Hyperparameters' column to dictionaries
data['Hyperparameters'] = data['Hyperparameters'].apply(parse_hyperparameters)

# Normalize the dictionary to separate columns and handle 'None' as NaN
hp_df = pd.json_normalize(data['Hyperparameters'])
hp_df = hp_df.replace('None', pd.NA)

# Concatenate the new columns with the original DataFrame
resulting_data = pd.concat([data.drop('Hyperparameters', axis=1), hp_df], axis=1)

# Save the resulting DataFrame to a new CSV file
output_file_path = 'processed_file.csv'
resulting_data.to_csv(output_file_path, index=False)

print("File processed and saved as:", output_file_path)
