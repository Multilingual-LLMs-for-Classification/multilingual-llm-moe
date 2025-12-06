import pandas as pd
import os

def combine_csv_files(directory_path, output_filename, specific_filenames):
    """
    Reads a specific list of CSV files from a specified directory and
    combines them into a single CSV file.

    Args:
        directory_path (str): The path to the directory containing the CSV files.
        output_filename (str): The name for the resulting combined CSV file.
        specific_filenames (list): A list of the exact CSV filenames to combine.
    """
    
    # Construct the full paths for the specified files
    all_files = [os.path.join(directory_path, f) for f in specific_filenames]
    
    # Check if any files were specified
    if not specific_filenames:
        print(f"Error: No specific CSV filenames were provided to combine.")
        return

    print(f"Attempting to combine {len(all_files)} specific files:")
    # Loop through the list of filenames to display which files are being processed
    for f_name in specific_filenames:
        print(f"- {f_name}")
        
    # List to hold the data frames
    df_list = []
    
    # Loop through all file paths and read each one into a pandas DataFrame
    for filename in all_files:
        # Check if the file actually exists before trying to read it
        if not os.path.exists(filename):
            print(f"Warning: File not found at path {filename}. Skipping.")
            continue
            
        try:
            # Read the CSV file. Adjust encoding or separator if necessary (e.g., sep=';').
            df = pd.read_csv(filename)
            df_list.append(df)
            
        except Exception as e:
            # os.path.basename(filename) gives the file name part, useful for logs
            print(f"Warning: Could not read file {os.path.basename(filename)}. Skipping. Error: {e}")

    # Check if any data frames were successfully loaded
    if not df_list:
        print("Error: No data could be loaded from the specified CSV files.")
        return

    # Concatenate all data frames in the list
    # ignore_index=True resets the index of the combined DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)

    # Write the combined data frame to a new CSV file in the specified directory
    output_path = os.path.join(directory_path, output_filename)
    combined_df.to_csv(output_path, index=False) # index=False prevents writing the DataFrame index

    print("\n--- Success ---")
    print(f"Successfully combined {len(df_list)} files.")
    print(f"Output saved to: {output_path}")

if __name__ == '__main__':
    # Ensure pandas is installed: pip install pandas
    
    # --- USER INPUT SECTION ---
    # NOTE: You must edit these three lines with your actual paths and file names.
    
    # 1. The path to the folder containing the CSV files (e.g., './data' or '/Users/user/Documents/data')
    input_directory = '../data' 
    
    # 2. The desired name for the combined output file
    output_name = 'test.csv'
    
    # 3. The list of specific CSV files (which must be inside input_directory) to combine
    input_files_array = ['test_da.csv', 'test_en.csv', 'test_es.csv', 'test_pl.csv', 'test_tr.csv']
    
    # --------------------------
    
    # Call the main combination function
    combine_csv_files(input_directory, output_name, input_files_array)