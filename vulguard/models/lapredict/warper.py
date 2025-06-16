from vulguard.models.BaseWraper import BaseWraper
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
import pickle, os
import pandas as pd

class LAPredict(BaseWraper):
    def __init__(self, language):        
        self.model_name = 'lapredict'
        self.language = language
        self.initialized = False
        self.model = None
        self.columns = (["la"])
        self.default_input = "Kamei_features"
                
    def initialize(self, **kwarg):
        model_path = kwarg.get("model_path")
        if model_path is None:
            self.model = sk_LogisticRegression(class_weight='balanced', max_iter=1000)
        else:
            self.model = pickle.load(open(f"{model_path}/la.pkl", "rb"))

        self.initialized = True
    
    def preprocess(self, data_df):
        print(f"Load data: {data_df}")
        data = pd.read_json(data_df, orient="records", lines=True)         
        
        commit_ids = data.loc[:, "commit_id"]
        features = data.loc[:, self.columns]
        labels = data.loc[:, "label"] if "label" in data.columns else None
           
        return commit_ids, features, labels
    
    def postprocess(self, commit_ids, outputs, threshold, labels=None, **kwargs):
        result = pd.DataFrame({
            "commit_id": commit_ids,
            "probability": outputs,
        })
        result["prediction"] = (result["probability"] > threshold).astype(float)
        
        if labels is not None:
            result["label"] = labels

        return result
    
    def inference(self, infer_df, threshold, **kwarg):
        commit_ids, features, labels = self.preprocess(infer_df)
        outputs = self.model.predict_proba(features)[:, 1]
        final_prediction = self.postprocess(commit_ids, outputs, threshold, labels)
        
        return final_prediction
    
    def train(self, **kwarg):
        train_df = kwarg.get("train_df")
        save_path = kwarg.get("save_path")
        
        _ , data, label = self.preprocess(train_df)
        assert label is not None, "Ensure there is label column in training data"
        self.model.fit(data, label)   
        self.save(save_path)          
    
    def save(self, save_path, **kwarg):
        os.makedirs(save_path, exist_ok=True)        
        save_path = f"{save_path}/la.pkl"
        pickle.dump(self.model, open(save_path, "wb"))