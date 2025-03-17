from defectguard.models.BaseWraper import BaseWraper
import pickle, os
from defectguard.utils.utils import download_folder, SRC_PATH
import pandas as pd

class LogisticRegression(BaseWraper):
    def __init__(self, language):
        self.model_name = 'lr'
        self.language = language
        self.initialized = False
        self.model = None
        self.columns = (["ns","nd","nf","entrophy","la","ld","lt","fix","ndev","age","nuc","exp","rexp","sexp"])
        download_folder(self.model_name, self.language)
        
    def initialize(self, pretrain=None):
        if pretrain:
            self.model = pickle.load(open(pretrain, "rb"))
        else:
            self.model = pickle.load(open(f"{SRC_PATH}/models/metadata/{self.model_name}/{self.language}", "rb"))

        # Set initialized to True
        self.initialized = True

    def preprocess(self, data):
        if not self.initialized:
            self.initialize()
        print("Preprocessing...")

        df = pd.DataFrame(data['features'])
        commit_ids = df["_id"].to_list()
        df = df.loc[:, self.columns]

        return commit_ids, df

    def inference(self, model_input):
        if not self.initialized:
            self.initialize()

        output = self.model.predict_proba(model_input)[:, 1]
        return output

    def postprocess(self, commit_ids, inference_output):
        if not self.initialized:
            self.initialize()

        result = []
        for commit_id, output in zip(commit_ids, inference_output):
            json_obj = {'commit_hash': commit_id, 'predict': output}
            result.append(json_obj)

        return result

    def handle(self, data):
        if not self.initialized:
            self.initialize()

        commit_ids, preprocessed_data = self.preprocess(data)
        model_output = self.inference(preprocessed_data)
        final_prediction = self.postprocess(commit_ids, model_output)

        return final_prediction
    
    def save(self, save_dir):
        if not os.path.isdir(save_dir):       
            os.makedirs(save_dir)
        
        save_path = f"{save_dir}/lr.pkl"
        pickle.dump(self.model, open(save_path, "wb"))

    def predict_proba(self, test):
        if not self.initialized:
            self.initialize()
            
        return self.model.predict_proba(test)