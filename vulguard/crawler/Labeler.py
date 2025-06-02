import pandas as pd
import glob
import os
from tqdm import tqdm
import json

class Labeler:
    def __init__(self, params):
        self.repo_name = params.repo_name
        self.save_path = params.save_path
        self.szz = params.szz if params.szz is not None else None
        
        self.label_file = f"{self.save_path}/vic_{self.szz}_{self.repo_name}.jsonl"
        self.commit_list = f"{self.save_path}/commit_ids_{self.repo_name}.jsonl"
        
    def run(self):
        data = self.link_label()
        train, val, test = self.split(data)  
        self.link_data(train, val, test)
          
    def link_label(self):
        VIC = []
        with open(self.label_file, "r") as f:
            for line in f:
                commit = json.loads(line.strip())
                for vic in commit["VIC"]:
                    VIC.append({
                        "commit_id": vic,
                        "label": 1
                    })
        label_df = pd.DataFrame(VIC)
                       
        commit_list_df = pd.read_json(self.commit_list, orient="records", lines=True)
        merged = commit_list_df.merge(label_df, on='commit_id', how='left')
        merged = merged.drop_duplicates(subset="commit_id", keep="first")

        merged['label'] = merged['label'].fillna(0).astype(int)
        save_file = f"{self.save_path}/labeled_commit_ids_{self.repo_name}.jsonl"
        merged.to_json(save_file, orient="records", lines=True)
        
        return merged
    
    def link_data(self, train, val, test):
        data_files = glob.glob(f"{self.save_path}/extracted/*.jsonl")
        data_dfs = [pd.read_json(data_file, orient="records", lines=True) for data_file in data_files]
        
        dataset_path = f"{self.save_path}/data"
        os.makedirs(dataset_path, exist_ok=True)
        
        for data_file, data_df in zip(data_files, data_dfs):
            filename = os.path.basename(data_file)
            
            if 'date' in train.columns and 'date' in data_df.columns:
                data_df = data_df.drop(columns=['date'])
                
            train_df = train.merge(data_df, on='commit_id', how='inner')
            train_df.to_json(f"{dataset_path}/train_{filename}", orient="records", lines=True)
            
            val_df = val.merge(data_df, on='commit_id', how='inner')
            val_df.to_json(f"{dataset_path}/val_{filename}", orient="records", lines=True)

            test_df = test.merge(data_df, on='commit_id', how='inner')
            test_df.to_json(f"{dataset_path}/test_{filename}", orient="records", lines=True)
  
    def split(self, df, label_ratio=(0.75, 0.05, 0.20)):
        assert sum(label_ratio) == 1.0, "Label ratio must sum to 1.0"

        # Step 1: Order all commits by date
        df = df.sort_values("date").reset_index(drop=True)

        # Step 2: Get all label 1 commits
        label1_df = df[df["label"] == 1].reset_index(drop=True)
        total_label1 = len(label1_df)

        # Step 3: Choose anchor commits
        n_train = int(label_ratio[0] * total_label1)
        n_val = int(label_ratio[1] * total_label1)

        anchor1 = label1_df.iloc[n_train]     # First commit in val
        anchor2 = label1_df.iloc[n_train + n_val - 1]  # Last commit in val

        # Get anchor dates
        anchor1_date = anchor1["date"]
        anchor2_date = anchor2["date"]

        # Step 4: All label 0 commits with date < anchor1 -> train
        label0_df = df[df["label"] == 0]
        train0_df = label0_df[label0_df["date"] < anchor1_date]
        test0_df = label0_df[label0_df["date"] > anchor2_date]
        val0_df = label0_df[
            (label0_df["date"] >= anchor1_date) & (label0_df["date"] <= anchor2_date)
        ]

        # Step 5: Split label 1 commits
        train1_df = label1_df.iloc[:n_train]
        val1_df = label1_df.iloc[n_train:n_train + n_val]
        test1_df = label1_df.iloc[n_train + n_val:]

        # Step 6: Combine and sort each split
        train_df = pd.concat([train1_df, train0_df]).sort_values("date").reset_index(drop=True)
        train_df = train_df.drop_duplicates(subset="commit_id", keep="first")

        val_df = pd.concat([val1_df, val0_df]).sort_values("date").reset_index(drop=True)
        val_df = val_df.drop_duplicates(subset="commit_id", keep="first")
        
        test_df = pd.concat([test1_df, test0_df]).sort_values("date").reset_index(drop=True)
        test_df = test_df.drop_duplicates(subset="commit_id", keep="first")

        return train_df, val_df, test_df

        