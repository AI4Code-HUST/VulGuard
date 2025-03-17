import pandas as pd
from imblearn.over_sampling import (
    RandomOverSampler, SMOTE, KMeansSMOTE
)
from imblearn.under_sampling import (
    RandomUnderSampler, OneSidedSelection, TomekLinks, NearMiss
)
from imblearn.combine import (
    SMOTETomek
)
from sklearn.cluster import MiniBatchKMeans
from .utils.utils import *


def sample(input_dataframe: pd.DataFrame, sample_strat, random_state:int = 42) -> pd.DataFrame:
    ids = df["commit_id"]
    X = df.drop(columns=["label", "commit_id"]) if sample_strat not in ["rus", "ros"] else df.drop(columns=["label"])
    y = df["label"]

    if sample_strat == "ros":
        sampler = RandomOverSampler(random_state=random_state)
    elif sample_strat == "rus":
        sampler = RandomUnderSampler(random_state=random_state)
    elif sample_strat == "smote":
        sampler = SMOTE(random_state=random_state)
    elif sample_strat == "oss":
        sampler = OneSidedSelection(random_state=random_state)
    elif sample_strat == "tomeklinks":
        sampler = TomekLinks()
    elif sample_strat == "SMOTETomek":
        sampler = SMOTETomek(random_state=random_state)
    elif sample_strat == "SMOTETomekv1":
        sampler = NearMiss(version=1)

    X_sampled, y_sampled = sampler.fit_resample(X, y)
    if sample_strat not in ["ros", "rus"]:
        df_sampled = pd.concat([ids, X_sampled, y_sampled], axis=1)
    else:
        df_sampled = pd.concat([X_sampled, y_sampled], axis=1)
    cols = X.columns.tolist()
    cols.append("label")
    df_sampled = df_sampled.dropna(subset=cols)
    df['commit_id'] = df['commit_id'].fillna('generated')
    return df_sampled

# strat = "rus"
# input_file_paths = [
#     "output/dataset/FFmpeg/{}/unsampling/{}-FFmpeg-features-train.jsonl",
#     "output/dataset/FFmpeg/{}/unsampling/{}-FFmpeg-simcom-train.jsonl",
#     "output/dataset/FFmpeg/{}/unsampling/{}-FFmpeg-deepjit-train.jsonl",
#     "output/dataset/FFmpeg/{}/unsampling/{}-FFmpeg-vcc-features-train.jsonl",
#     # "output/dataset/FFmpeg/SETUP5/unsampling/SETUP5-FFmpeg-simcom-train.jsonl",
#     # "output/dataset/FFmpeg/SETUP5/unsampling/SETUP5-FFmpeg-deepjit-train.jsonl",
# ]

# output_file_paths = [
#     "output/dataset/FFmpeg/{}/{}/{}-FFmpeg-features-train.jsonl",
#     "output/dataset/FFmpeg/{}/{}/{}-FFmpeg-simcom-train.jsonl",
#     "output/dataset/FFmpeg/{}/{}/{}-FFmpeg-deepjit-train.jsonl",
#     "output/dataset/FFmpeg/{}/{}/{}-FFmpeg-vcc-features-train.jsonl",
#     # "output/dataset/FFmpeg/SETUP5/smote/SETUP5-FFmpeg-simcom-train.jsonl",
#     # "output/dataset/FFmpeg/SETUP5/smote/SETUP5-FFmpeg-deepjit-train.jsonl",
# ]

# paths = [
#     f"output/dataset/FFmpeg/SETUP1/{strat}",
#     f"output/dataset/FFmpeg/SETUP2/{strat}",
#     f"output/dataset/FFmpeg/SETUP3/{strat}",
#     f"output/dataset/FFmpeg/SETUP4/{strat}",
#     f"output/dataset/FFmpeg/SETUP5/{strat}",
# ]

# for path in paths:
#     if not os.path.exists(path):
#         os.makedirs(path)
# for input, output in zip(input_file_paths, output_file_paths):
#     for setup in ["SETUP1", "SETUP2", "SETUP3", "SETUP4", "SETUP5"]:
#         data = list()
#         input_file = input.format(setup, setup)
#         output_file = output.format(setup, strat, setup)
#         for line in load_jsonl(input_file):
#             data.append(line)

#         df = pd.DataFrame(data)
#         df_sampled = sample(df, strat)
#         data = df_sampled.to_dict(orient = "records")
#         save_jsonl(data, output_file)
        
#         print(f"{input_file} to {output_file}")