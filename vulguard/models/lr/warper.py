from vulguard.models.BaseWraper import BaseWraper
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
import pickle, os
from vulguard.utils.utils import SRC_PATH
import pandas as pd

class LogisticRegression(BaseWraper):
    def __init__(self, language):        
        self.model_name = 'lr'
        self.language = language
        self.initialized = False
        self.model = None
        self.columns = (["ns","nd","nf","entropy","la","ld","lt","fix","ndev","age","nuc","exp","rexp","sexp"])
        self.default_input = "Kamei_features"
        
    def initialize(self, **kwarg):
        model_path = kwarg.get("model_path")
        if model_path is None:
            self.model = sk_LogisticRegression(class_weight='balanced', max_iter=1000) 
        else:
            self.model = pickle.load(open(f"{model_path}/lr.pkl", "rb"))
            
        self.initialized = True
        
    def preprocess(self, data, **kwarg):         
        commit_ids = data.loc[:, "commit_id"]
        features = data.loc[:, self.columns]
        labels = data.loc[:, "label"] if "label" in data.columns else None

        return commit_ids, features, labels
    
    def postprocess(self, commit_ids, outputs, threshold, **kwarg):
        result = []
        for commit_id, output in zip(commit_ids, outputs):
            json_obj = {
                'commit_id': commit_id, 
                'probability': output, 
                'prediction': float(output > threshold)
            }
            result.append(json_obj)
        result = pd.DataFrame(result)
        return result
    
    def inference(self, infer_df, threshold, **kwarg):        
        commit_ids, features, _ = self.preprocess(infer_df)
        outputs = self.model.predict_proba(features)[:, 1]
        final_prediction = self.postprocess(commit_ids, outputs, threshold)
        
        return final_prediction
    
    def train(self, **kwarg):
        train_df = kwarg.get("train_df")
        _ , data, label = self.preprocess(train_df)
        self.model.fit(data, label)        
    
    def save(self, save_path, **kwarg):
        os.makedirs(save_path, exist_ok=True)        
        save_path = f"{save_path}/lr.pkl"
        pickle.dump(self.model, open(save_path, "wb"))