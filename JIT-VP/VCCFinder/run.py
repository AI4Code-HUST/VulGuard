import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef
)
from scipy.sparse import csr_matrix
import pickle
  
def get_commit_id_list(file):
    commit_id_list = list()
    with open(file, "r") as input_file:
        all_lines = input_file.readlines()
    for line in all_lines:
        dirty_id = (line.split('#'))[1].replace('\n', '')
        commit_id_list.append(dirty_id)
    return commit_id_list
   
def load_data(train_file, test_file):
    print("\tStart load")
    (Xgt_train, Ygt_train) = load_svmlight_file(train_file, dtype=bool)
    #Ground Truth Test Data
    (Xgt_test, Ygt_test) = load_svmlight_file(test_file, dtype=bool)
    
    max_features_all_files = max( Xgt_train.shape[1], Xgt_test.shape[1])

    Xgt_train = csr_matrix( Xgt_train, shape=(Xgt_train.shape[0], max_features_all_files ) )
    Xgt_test = csr_matrix( Xgt_test, shape = (Xgt_test.shape[0], max_features_all_files ) )
    print("\tDone load")
    return Xgt_train, Ygt_train, Xgt_test, Ygt_test

def train(Xgt_train, Ygt_train):
    print("\tStart train")
    params_weighted={"max_iter":200000,"class_weight":{0: 1,1: 100}}
    classif = LinearSVC()
    classif.set_params(**params_weighted)
    classif.fit(Xgt_train, Ygt_train)
    print("\tDone train")
    
    return classif

def test(classif, Xgt_test, Ygt_test, test_file):
    print("\tStart eval")
    preds_test = classif.predict(Xgt_test)
    probas_test = classif.decision_function(Xgt_test)

    commit_id = get_commit_id_list(test_file)
    out_df = pd.DataFrame(zip(commit_id, Ygt_test, probas_test, preds_test), columns=["commit_hash", "label", "proba", "pred"])
    
    f1 = f1_score(y_true=Ygt_test, y_pred=preds_test)
    mcc = matthews_corrcoef(y_true=Ygt_test, y_pred=preds_test)
    
    roc_auc = roc_auc_score(y_true=Ygt_test,  y_score=probas_test)
    precision, recall, _ = precision_recall_curve(y_true=Ygt_test, probas_pred=probas_test)
    pr_auc = auc(recall, precision)
    
    score_df = pd.DataFrame([[roc_auc, pr_auc, f1, mcc]], columns= ["roc_auc", "pr_auc", "f1", "mcc"])
    print("\tDone eval")
    return out_df, score_df

def run(train_file, test_file, mode):
    if mode not in ["train_only", "test_only", "all"]:
        print("Specify the mode!")
        return None
    
    Xgt_train, Ygt_train, Xgt_test, Ygt_test = load_data(train_file, test_file)
    if mode != "test_only":
        classif = train(Xgt_train, Ygt_train)
    if mode != "train_only":
        out_df, score_df = test(classif, Xgt_test, Ygt_test)
        

    return classif, out_df, score_df
