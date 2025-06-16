from vulguard.models.BaseWraper import BaseWraper
from sklearn.datasets import load_svmlight_file
from sklearn.svm import LinearSVC
from scipy.sparse import csr_matrix
import pickle, os
import pandas as pd
from .dataset import prep_svm_data

class VCCFinder(BaseWraper):
    def __init__(self, language):        
        self.model_name = 'vccfinder'
        self.language = language
        self.initialized = False
        self.model = None
        self.columns = ([
            "addition", "deletion", "hunk_count", "kw_do", "kw_if", "kw_asm", "kw_for", "kw_int", "kw_new", "kw_try", "kw_auto", "kw_bool", "kw_case", "kw_char",
            "kw_else", "kw_enum", "kw_free", "kw_goto", "kw_long", "kw_this", "kw_true", "kw_void", "kw_alloc", "kw_break", "kw_catch", "kw_class", "kw_const", 
            "kw_false", "kw_float", "kw_short", "kw_throw", "kw_union", "kw_using", "kw_while", "kw_alloca", "kw_calloc", "kw_delete", "kw_double","kw_extern",
            "kw_friend", "kw_inline", "kw_malloc", "kw_public", "kw_return", "kw_signed", "kw_sizeof", "kw_static", "kw_struct", "kw_switch", "kw_typeid", "kw_default",
            "kw_mutable", "kw_private", "kw_realloc", "kw_typedef", "kw_virtual", "kw_wchar_t", "kw_continue", "kw_explicit", "kw_operator", "kw_register", "kw_template",
            "kw_typename", "kw_unsigned", "kw_volatile", "kw_namespace", "kw_protected", "kw_const_cast", "kw_static_cast", "kw_dynamic_cast", "kw_reinterpret_cast",
            "author_contributions_percent", "past_changes", "future_changes", "past_different_authors", "future_different_authors"
        ])
        self.num_features = None 
        self.default_input = "VCC_features,patch"
        
    def initialize(self, **kwarg):
        model_path = kwarg.get("model_path")
        if model_path is None:
            params_weighted= {
                "max_iter":200000,
                "class_weight":{0: 1,1: 100}
            }
            self.model = LinearSVC()
            self.model.set_params(**params_weighted) 
        else:
            self.model = pickle.load(open(f"{model_path}/vccfinder.pkl", "rb"))
            self.num_features = self.model.coef_.shape[1]
            
        self.initialized = True
        
    def preprocess(self, data_df):
        print(f"Load data: {data_df}")
        feature_path, patch_path = data_df.split(",")
        
        feature_df = pd.read_json(feature_path, orient="records", lines=True)
        patch_df = pd.read_json(patch_path, orient="records", lines=True)
        feature_df["messages"] = patch_df["messages"]
        
        commit_ids = feature_df.loc[:, "commit_id"]
        labels = patch_df.loc[:, "label"]
          
        directory = os.path.dirname(feature_path)
        basename = os.path.splitext(os.path.basename(feature_path))[0]
        output_svm_file = f"{directory}/{basename}.libsvm" 
        
        if not os.path.exists(output_svm_file):
            return_code = prep_svm_data(feature_df, output_svm_file)
            if not return_code:
                exit()
        
        (features, _) = load_svmlight_file(output_svm_file, dtype=bool)
        num_features = features.shape[1] if  self.num_features is None else self.num_features 
        features = csr_matrix( features, shape=(features.shape[0], num_features ) )
        
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
        params = kwarg.get("params")
        threshold = 0 if params.threshold is not None else params.threshold       
        commit_ids, features, labels = self.preprocess(infer_df)
        outputs = self.model.decision_function(features)
        final_prediction = self.postprocess(commit_ids, outputs, threshold, labels)
        
        return final_prediction
    
    def train(self, **kwarg):
        train_df = kwarg.get("train_df")
        save_path = kwarg.get("save_path")
        
        _ , data, label = self.preprocess(train_df)
        self.model.fit(data, label)   
        self.save(save_path)     
    
    def save(self, save_path, **kwarg):
        os.makedirs(save_path, exist_ok=True)        
        save_path = f"{save_path}/vccfinder.pkl"
        pickle.dump(self.model, open(save_path, "wb"))