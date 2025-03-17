from icecream import ic as logger
from datetime import datetime
import pandas as pd
import subprocess, os

logger.configureOutput(prefix=f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | ')

def logs(file_path, project_name, value, model):
    # Read existing CSV file
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=["Project Name", "lapredict", "lr", "tlel", "deepjit", "sim", "com", "simcom"])
    
    # Append new data to DataFrame
    filtered_df = df[df["Project Name"] == project_name]
    
    # Update the cell corresponding to 'CC2Vec' column with the AUC score
    if not filtered_df.empty:
        df.at[filtered_df.index[0], model] = value
        # If you expect multiple rows with the same project_name, you may want to iterate over filtered_df to update all corresponding rows
    else:
        # If filtered DataFrame is empty, create a new row
        new_row = {"Project Name": project_name, "lapredict": 0.0, "lr": 0.0, "tlel": 0.0, "deepjit": 0.0, "sim": 0.0, "com": 0.0, "simcom": 0.0}
        new_row[model] = value
        df = df._append(new_row, ignore_index=True)

    # Write DataFrame back to CSV file
    df.to_csv(file_path, index=False)

def get_vram_usage():
    # Get GPU memory usage using nvidia-smi command
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits'])
        vram_usage = int(output.strip().split(b'\n')[1])
        return vram_usage
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None  # Handle case where nvidia-smi is not available or GPU is not Nvidia