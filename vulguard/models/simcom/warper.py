from vulguard.models.BaseWraper import BaseWraper
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
        self.default_input = "Kamei_features,patch"
        
    def initialize(self, dictionary, hyperparameters, model_path=None, **kwarg):
        self.sim.initialize(model_path=model_path)
        self.com.initialize(dictionary=dictionary, hyperparameters=hyperparameters, model_path=model_path)
        self.initialized = True
        
    def preprocess(self, path):
        sim_path, com_path = path.split(",")
        return sim_path, com_path

    def postprocess(self, sim_predict, com_predict, threshold):
        columns = sim_predict.columns
        final_predict = pd.merge(sim_predict, com_predict, on='commit_id', suffixes=('_1', '_2'))
        final_predict['probability'] = (final_predict['probability_1'] + final_predict['probability_2']) / 2
        final_predict['prediction'] = (final_predict['probability'] > threshold).astype(float)
        if "label" in columns:
            final_predict["label"] = sim_predict["label"]
        return final_predict[columns]

    def inference(self, infer_df, threshold, **kwarg):
        sim_infer, com_infer = self.preprocess(infer_df)
        
        print("Infer Sim:")
        sim_predict = self.sim.inference(infer_df=sim_infer, threshold=threshold, **kwarg)        
        print("Infer Com:")
        com_predict = self.com.inference(infer_df=com_infer, threshold=threshold, **kwarg)

        final_predict = self.postprocess(sim_predict, com_predict, threshold)
        return final_predict

    def train(self, train_df, val_df, **kwarg):
        sim_train, com_train = self.preprocess(train_df)
        _ , com_val = self.preprocess(val_df)

        print("Train Sim:")
        self.sim.train(sim_train, **kwarg)        
        print("Train Com:")
        self.com.train(com_train, com_val, **kwarg)

    def save(self, save_path, **kwarg):
        os.makedirs(save_path, exist_ok=True)        
        self.sim.save(save_path=save_path)
        self.com.save(save_path=save_path)