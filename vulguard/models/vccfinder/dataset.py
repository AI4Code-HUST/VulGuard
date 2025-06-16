import pandas as pd
import numpy as np
import json, os, shutil
import subprocess
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer


def prep_svm_data(df, output_svm_file):
    print("Prepare svm input!")
    df = df.copy()
    if "label" not in df.columns:
        df["label"] = 0

    temp_file = f"{os.path.dirname(output_svm_file)}/oneline.libsvm"
    # === Step 2: Scale numeric fields except 'id' and 'label' ===
    exclude_fields = {"commit_id", "date", "author", "files", "label", "messages"}
    required_fields = {"label", "messages"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(exclude_fields)

    # Apply QuantileTransformer
    scaler = QuantileTransformer(output_distribution="uniform")
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # === Step 3: Convert DataFrame to JSONL string ===
    filtered_df = df[list(numeric_cols) + list(required_fields)]
    # print(filtered_df)
    # exit()
    jsonl_data = filtered_df.to_dict(orient="records")
    jsonl_strings = [json.dumps(record) for record in jsonl_data]
    

    # === Step 4: Call Sally using subprocess (stdin -> Sally -> .svm) ===
    #"sally -i lines -o libsvm --vect_embed bin -d' ' -g tokens " + WORKSPACE + "/* " + WORKSPACE + "/one_line.libsvm"
    open(output_svm_file, 'w').close() 
    for jsonl_string in tqdm(jsonl_strings):
        sally_cmd = [
            "sally",
            "-i", "stdin",
            "-o", "libsvm",
            "--vect_embed", "bin",
            "-d", "\' \'",
            "-g", "tokens",
            jsonl_string,
            temp_file
        ]


        process = subprocess.Popen(
            sally_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # use text mode for string input
        )

        _ , stderr = process.communicate(input=jsonl_string)
                
    # === Step 5: Handle subprocess output ===
        if process.returncode != 0:
            print("Sally error:\n", stderr)
            return 0
        else:
            with open(temp_file, 'r') as tf, open(output_svm_file, 'a') as af:
                shutil.copyfileobj(tf, af)
    
    os.remove(temp_file)
    print(f"Embedding complete. Output written to {output_svm_file}")
    return 1