import json, torch, os
import pandas as pd
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import  AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaTokenizer, RobertaModel

from vulguard.models.BaseWraper import BaseWraper

from .model import Model
from .dataset import TextDataset

from sklearn.metrics import precision_recall_curve, auc

def get_auc(ground_truth, predict):
    precisions, recalls, _ = precision_recall_curve(y_true=ground_truth, probas_pred=predict)
    pr_auc = auc(recalls, precisions)
    
    return pr_auc

class JITFine(BaseWraper):
    def __init__(self, language, device="cpu", **kwarg):
        self.model_name = 'jitfine'
        self.language = language
        self.initialized = False
        self.model = None
        self.device = device
        self.hyperparameters = None 

        self.optimizer = None
        self.scheduler = None
        self.epoch = 0
        self.step = 0
        self.patience = 0  
        
        self.config = None
        self.tokenizer = None
        self.encoder = None
        
        self.val_dataloader = None
                
        self.default_input = "Kamei_features,merge"

    def __call__(self, message, code):
        return self.model(message, code)
    
    def get_parameters(self):
        return self.model.parameters()
    
    def set_device(self, device):
        self.device = device
    
    def initialize(self, hyperparameters, model_path=None, **kwarg):        
        # Load hyperparameter
        with open(hyperparameters, 'r') as file:
            self.hyperparameters = json.load(file)
            
        # Init config
        self.config = RobertaConfig.from_pretrained(self.hyperparameters["config_name"] if self.hyperparameters["config_name"] else self.hyperparameters["model_name_or_path"])    
        self.config.num_labels = self.hyperparameters["num_labels"]
        self.config.feature_size = self.hyperparameters["feature_size"]
        self.config.hidden_dropout_prob = self.hyperparameters["head_dropout_prob"]

        # Init tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(self.hyperparameters["tokenizer_name"])
        special_tokens_dict = {'additional_special_tokens': ["<ADD>", "<REMOVE>"]}
        self.tokenizer.add_special_tokens(special_tokens_dict)

        # Init encoder
        self.encoder = RobertaModel.from_pretrained(self.hyperparameters["model_name_or_path"], config=self.config)    
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        
        # Init model
        self.model = Model(self.encoder, self.config, self.tokenizer, self.hyperparameters).to(device=self.device)
        
        # Init optimizer
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.hyperparameters["weight_decay"]
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 
                'weight_decay': 0.0
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.hyperparameters["train"]["learning_rate"], eps=self.hyperparameters["adam_epsilon"])
        
        
        if model_path is not None:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=1,
                num_training_steps=1
            )
            
            checkpoint = torch.load(f"{model_path}/jitfine.pth")  # Load the last saved checkpoint
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.epoch = checkpoint['epoch']
            self.step = checkpoint['step']
            self.patience = checkpoint["patience"]

        # Set initialized to True
        self.initialized = True
    
    def preprocess(self, data_path, mode="train", **kwarg):
        feature_path, code_path = data_path.split(",")
        
        dataset = TextDataset(tokenizer=self.tokenizer, hyperparameters=self.hyperparameters, changes_filename=code_path, features_filename=feature_path, mode=mode)
        if mode == "train":
            sampler = RandomSampler(dataset)
        elif mode == "test" or mode == "val":
            sampler = SequentialSampler(dataset)
        
        return dataset, sampler
    
        
    def postprocess(self, commit_ids, outputs, threshold, labels=None, **kwargs):
        result = pd.DataFrame({
            "commit_id": commit_ids,
            "probability": outputs,
        })
        result["prediction"] = (result["probability"] > threshold).astype(float)
        
        if labels is not None:
            result["label"] = labels

        return result

    def inference(self, infer_df, threshold, eval_dataloader= None, **kwarg):
        if eval_dataloader is None:
            eval_dataset, eval_sampler = self.preprocess(infer_df, mode="test", **kwarg)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.hyperparameters["test"]["eval_batch_size"])
        
        # Eval!
        print("***** Running evaluation *****")
        # print("  Num examples = %d", len(eval_dataset))
        # print("  Batch size = %d", self.hyperparameters[mode]["eval_batch_size"])

        eval_loss = 0.0
        self.model.eval()
        commit_ids = []
        logits = []
        y_trues = []
        
        bar = tqdm(eval_dataloader, total=len(eval_dataloader))
        for batch in bar:
            commit_id, input_ids, attn_masks, manual_features, labels = batch
            input_ids = input_ids.to(self.device)
            attn_masks = attn_masks.to(self.device)
            manual_features = manual_features.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                if labels is not None:
                    loss, logit, _ = self.model(input_ids, attn_masks, manual_features, labels)
                    eval_loss += loss.mean().item()
                else:
                    logit = self.model(input_ids, attn_masks, manual_features, labels)
                    
                torch.cuda.empty_cache()
                logits.append(logit.cpu().numpy())
                commit_ids.append(commit_id)
                y_trues.append(labels.cpu().numpy() if labels is not None else None)
            bar.update()
            
        # calculate scores
        logits = np.concatenate(logits, 0)
        commit_ids = np.concatenate(commit_ids, 0)
        y_trues = np.concatenate(y_trues, 0)
        y_trues = y_trues if y_trues[0] is not None else None
        
        final_prediction = self.postprocess(commit_ids=commit_ids, outputs=logits, threshold=threshold, labels=y_trues)
        # print(final_prediction)
        # print(commit_ids)
        return final_prediction
        
    def train(self, train_df, val_df, **kwarg):
        params = kwarg.get("params")
        save_path = kwarg.get("save_path")
        threshold = 0.5 if params.threshold is None else params.threshold
        
        train_dataset, train_sampler = self.preprocess(train_df, mode="train", **kwarg)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.hyperparameters["train"]["train_batch_size"])
        
        val_dataset, val_sampler = self.preprocess(val_df, mode="val", **kwarg)
        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=self.hyperparameters["train"]["eval_batch_size"])
        val_ground_truth = [ example.label for example in val_dataset.examples if example.label is not None ]
        # print(val_ground_truth)
        
        if self.hyperparameters["max_steps"] > 0:
            max_steps =  self.hyperparameters["max_steps"]
        else:
            max_steps = self.hyperparameters["epochs"] * len(train_dataloader) if params.epochs is None else params.epochs * len(train_dataloader)
        
        save_steps = max(len(train_dataloader) // 5, 1)
        warmup_steps = self.hyperparameters["warmup_steps"]
        total_epochs = self.hyperparameters["epoches"] if params.epochs is None else params.epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        
        if self.scheduler is None:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps
            )

        # Train!
        print("***** Running training *****")
        print("  Num examples = ", len(train_dataset))
        print("  Num Epochs = ", total_epochs)
        print("  Instantaneous batch size per GPU = ", self.hyperparameters["train"]["train_batch_size"])
        print("  Total train batch size = ", self.hyperparameters["train"]["train_batch_size"] * self.hyperparameters["gradient_accumulation_steps"])
        print("  Gradient Accumulation steps = ", self.hyperparameters["gradient_accumulation_steps"])
        print("  Total optimization steps = ", max_steps)

        best_pr_auc = 0
        global_step = 0
        self.model.zero_grad()

        for idx in range(total_epochs):
            bar = tqdm(train_dataloader, total=len(train_dataloader))
            tr_loss = 0
            tr_num = 0
            for step, batch in enumerate(bar):
                commit_id, input_ids, attn_masks, manual_features, labels = batch
                input_ids = input_ids.to(self.device)
                attn_masks = attn_masks.to(self.device)
                manual_features = manual_features.to(self.device)
                labels = labels.to(self.device)
                self.model.train()
                loss, logits, _ = self.model(input_ids, attn_masks, manual_features, labels)
                
                if self.hyperparameters["gradient_accumulation_steps"] > 1:
                    loss = loss / self.hyperparameters["gradient_accumulation_steps"]

                # report loss
                tr_loss += loss.item()
                tr_num += 1
                if (step + 1) % save_steps == 0:
                    print("Epoch {} step {} loss {}".format(idx, step + 1, round(tr_loss / tr_num, 5)))
                    tr_loss = 0
                    tr_num = 0

                # backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyperparameters["max_grad_norm"])

                if (step + 1) % self.hyperparameters["gradient_accumulation_steps"] == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                global_step += 1
                if (step + 1) % save_steps == 0:
                    prediction = self.inference(infer_df=val_df, threshold=threshold, eval_dataloader= val_dataloader,**kwarg)
                    val_predict = prediction.loc[:, "probability"]
            
                    pr_auc = get_auc(val_ground_truth, val_predict)
                    # Save model checkpoint
                    if pr_auc > best_pr_auc:
                        best_pr_auc = pr_auc
                        print("  " + "*" * 20)
                        print("  Best pr_auc: %s", round(best_pr_auc, 4))
                        print("  " + "*" * 20)
                        self.patience = 0
                        
                        # save
                        self.save(f"{save_path}")
                        print(f"Saving best model checkpoint to {save_path}")
                    else:
                        self.patience += 1
                        if self.patience > self.hyperparameters["patience"] * 5:
                            print('Patience greater than {}, early stop!'.format(5 * self.patience))
                            return
                bar.update()            
        
        self.save(f"{save_path}")
        print(f"Saving last model checkpoint to {save_path}")
    
    def save(self, save_path, **kwarg):
        os.makedirs(save_path, exist_ok=True)
        
        save_path = f"{save_path}/jitfine.pth"
        torch.save({
            'epoch': self.epoch,
            'step': self.step,
            'patience': self.patience,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, save_path)