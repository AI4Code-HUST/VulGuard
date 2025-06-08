from vulguard.models.BaseWraper import BaseWraper
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
import pickle, os
from vulguard.utils.utils import SRC_PATH
import pandas as pd

class Sim(BaseWraper):
    def __init__(self, language, **kwarg):
        self.model_name = 'sim'
        self.language = language
        self.initialized = False
        self.model = RandomForestClassifier()
        self.columns = (["ns","nd","nf","entropy","la","ld","lt","fix","ndev","age","nuc","exp","rexp","sexp"])
        self.default_save = f"{SRC_PATH}/models/metadata/{self.model_name}/{self.language}"
        self.default_input = "Kamei_features"
        
    def initialize(self, model_path=None, **kwarg):
        if model_path is not None:
            self.model = pickle.load(open(f"{model_path}/sim.pkl", "rb"))
        else:
            self.model = pickle.load(open(f"{self.default_save}/sim.pkl", "rb"))

        # Set initialized to True
        self.initialized = True
        
    def preprocess(self, data, **kwarg):         
        commit_ids = data.loc[:, "commit_id"]
        features = data.loc[:, self.columns]
        labels = data.loc[:, "label"] if "label" in data.columns else None
        features, labels = RandomUnderSampler(random_state=42).fit_resample(features, labels)
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
        if not self.initialized:
            self.initialize()
        
        commit_ids, features, _ = self.preprocess(infer_df)
        outputs = self.model.predict_proba(features)[:, 1]
        final_prediction = self.postprocess(commit_ids, outputs, threshold)
        
        return final_prediction
    
    def train(self, train_df, val_df, **kwarg):
        commit_ids, data, label = self.preprocess(train_df)
        self.model.fit(data, label)        
        return self.model
    
    def save(self, save_dir, **kwarg):
        if not os.path.isdir(save_dir):       
            os.makedirs(save_dir)
        
        save_path = f"{save_dir}/sim.pkl"
        pickle.dump(self.model, open(save_path, "wb"))