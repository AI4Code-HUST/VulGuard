from vulguard.models.BaseWraper import BaseWraper
from vulguard.utils.utils import remove_file_from_path
from .sim.warper import Sim
from .com.warper import Com
import pandas as pd
import os

class SimCom(BaseWraper):
    def __init__(self, language, device="cpu"):
        self.model_name = 'simcom'
        self.language = language
        self.device = device
        self.initialized = False
        
        self.sim = Sim(self.language)
        self.com = Com(self.language, self.device)
        self.default_input = "Kamei_features"
        
    def initialize(self, dictionary, hyperparameters, model_path=None, **kwarg):
        self.sim.initialize(model_path=model_path)
        self.com.initialize(dictionary=dictionary, hyperparameters=hyperparameters, model_path=model_path)
        self.initialized = True
        
    def preprocess(self):
        pass

    def postprocess(self, sim_predict, com_predict, threshold):
        final_predict = pd.merge(sim_predict, com_predict, on='commit_id', suffixes=('_1', '_2'))
        final_predict['probability'] = (final_predict['probability_1'] + final_predict['probability_2']) / 2
        final_predict['prediction'] = (final_predict['probability'] > threshold).astype(float)
        
        return final_predict[['commit_id', 'probability', 'prediction']]

    def inference(self, infer_df, threshold, **kwarg):
        infer_path = remove_file_from_path(infer_df)
        params = kwarg.get("params")
        
        print("Infer Sim:")
        sim_infer = f'{infer_path}/test_{self.sim.default_input}_{params.repo_name}.jsonl' 
        sim_predict = self.sim.inference(infer_df=sim_infer, threshold=threshold, **kwarg)
        
        print("Infer Com:")
        com_infer = f'{infer_path}/test_{self.com.default_input}_{params.repo_name}.jsonl' 
        com_predict = self.com.inference(infer_df=com_infer, threshold=threshold, **kwarg)
        final_predict = self.postprocess(sim_predict, com_predict, threshold)
        return final_predict

    def train(self, train_df, val_df, **kwarg):
        train_path = remove_file_from_path(train_df)
        val_path = remove_file_from_path(val_df)
        params = kwarg.get("params")

        print("Train Sim:")
        sim_train = f'{train_path}/train_{self.sim.default_input}_{params.repo_name}.jsonl'
        self.sim.train(sim_train, **kwarg)
        
        print("Train Com:")
        com_train = f'{train_path}/train_{self.com.default_input}_{params.repo_name}.jsonl'
        com_val = f'{val_path}/val_{self.com.default_input}_{params.repo_name}.jsonl'
        self.com.train(com_train, com_val, **kwarg)

    def save(self, save_path, **kwarg):
        os.makedirs(save_path, exist_ok=True)        
        self.sim.save(save_path=save_path)
        self.com.save(save_path=save_path)