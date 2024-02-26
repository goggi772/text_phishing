import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, recall_score
from transformers import get_constant_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self,
                 seed=42,
                 train_loader=None,
                 valid_loader=None,
                 test_loader = None,
                 learning_rate=1e-6,
                 num_epochs=5,
                 warmup_steps=0,
                 outputs_dir:str = 'outputs',
                 logger_name=None,
                 device=None) -> None:
        

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.epochs = num_epochs
        self.learning_rate=learning_rate
        self.device = device
        
        self.warmup_steps = warmup_steps
        
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)
        self.outputs_dir = outputs_dir
        
        if logger_name is not None:
            self.writer = SummaryWriter(logger_name)
        
    def configure_optimizer(self, model):
        # optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.0005)
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
        
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
        # if self.warmup_steps >= 2:
        #     scheduler = get_constant_schedule_with_warmup(
        #         optimizer=optimizer,
        #         num_warmup_steps=self.warmup_steps
        #     )
        #     return optimizer, scheduler
        return optimizer, None
        # return optimizer, scheduler
    
    def accuracy(self, preds, labels):
        preds = np.argmax(preds, axis=-1)
        return accuracy_score(y_pred=preds, y_true=labels)
    
    def calculate_metric(self, preds, labels):
        acc = accuracy_score(y_pred=preds, y_true=labels)
        f1 = f1_score(y_pred=preds, y_true=labels, average='macro')
        recall = recall_score(y_pred=preds, y_true=labels, average='macro')
        
        return {'accuaracy': acc,
                'recall': recall,
                'f1': f1}
        
        
    
    def fit(self, model):
        best_valid_loss, best_valid_accuracy = 1e10, 0.
        record = defaultdict(list)
        optimizer, scheduler = self.configure_optimizer(model)

        for epoch in range(1, self.epochs+1):
            train_outputs = self.train(model, round=epoch-1, optimizer=optimizer)
            valid_outputs = self.evaluate(model, round=epoch-1)
            
            # scheduler.step()

            record['epoch'].append(epoch)
            
            record['train_loss'].append(train_outputs['loss'])
            record['train_accuracy'].append(train_outputs['accuracy'])
            record['valid_loss'].append(valid_outputs['loss'])
            record['valid_accuracy'].append(valid_outputs['accuracy'])

            if valid_outputs['loss'] < best_valid_loss:
                best_valid_loss = valid_outputs['loss']
                print("update valid loss: ", best_valid_loss)
                model.save_pretrained(os.path.join(self.outputs_dir, f'model_best_valid_loss'))

            if valid_outputs['accuracy'] > best_valid_accuracy:
                best_valid_accuracy = valid_outputs['accuracy']
                print("update valid accuracy: ", best_valid_accuracy)
                model.save_pretrained(os.path.join(self.outputs_dir, f'model_best_valid_accuracy'))

            print("=" * 64)
            print(f"[EPOCH] #{epoch}\n",
                  train_outputs, 
                  "\n",
                  valid_outputs)
            print("=" * 64)
            print("\n\n")

        df = pd.DataFrame(record)
        df.to_csv(os.path.join(self.outputs_dir, "train_valid_output.csv"), index=False)
    
    def train(self, model, round, optimizer):
        model.train()
        model.to(self.device)
        # optimizer, _ = self.configure_optimizer(model)

        progress_bar = tqdm(self.train_loader)

        total_loss = 0.
        total_acc = 0.
        
        length = len(self.train_loader)
        count = 0
        
        for x in progress_bar:
            x = {k:v.to(self.device) for k, v in x.items()}
            # augmented = {k + "_augmented": v.to(self.device) for k, v in x['augmented_data']}
            # output = model(**x)
            output = model(**x)

            loss = output['loss']
            logits = output['logits']
            
            # output2 = model(**augmented)
            
            # aug_loss = output2['loss']
            
            # loss = (ori_loss + aug_loss) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            current_loss = loss.cpu().item()
            total_loss += current_loss
            
            
            acc = self.accuracy(logits.cpu().detach().numpy(), x['labels'].cpu().detach().numpy())
            total_acc += acc
            
            self.writer.add_scalar('Train/loss', current_loss, (round * length) + count)
            self.writer.add_scalar('Train/acc', acc, (round * length) + count)
            
            progress_bar.set_description(f"[TRAIN] loss: {current_loss:.5f}, accuracy: {acc:.5f}|")
            count += 1

        return {'loss':total_loss / len(self.train_loader), 
                'accuracy': total_acc / len(self.train_loader)}
        
    def evaluate(self, model, round=0):
        model.eval()
        model.to(self.device)
        
        progress_bar = tqdm(self.valid_loader)

        total_loss, total_acc = 0., 0.
        length = len(self.valid_loader)
        count = 0
        
        for x in progress_bar:
            x = {k:v.to(self.device) for k, v in x.items()}
            with torch.no_grad():
                output = model(**x)

            loss = output['loss']
            logits = output['logits']            
            
            current_loss = loss.cpu().item()
            total_loss += current_loss
            
            acc = self.accuracy(logits.cpu().detach().numpy(), x['labels'].cpu().detach().numpy())
            total_acc += acc
            
            
            self.writer.add_scalar('Valid/loss', current_loss, (round * length) + count)
            self.writer.add_scalar('Valid/acc', acc, (round * length) + count)
            
            progress_bar.set_description(f"[VALID] loss: {current_loss:.5f}, accuracy: {acc:.5f}|")
            count += 1

        return {'loss':total_loss / length, 
                'accuracy': total_acc / length}
        
    def test(self, model):
        model.eval()
        model.to(self.device)
        
        progress_bar = tqdm(self.test_loader, desc='[TEST]')
        total_output = defaultdict(list)
        length = len(self.test_loader)
        total_logits, total_labels, indices = [], [], []
        miss_predict = []
        
        for batch_idx, x in enumerate(progress_bar):
            x = {k:v.to(self.device) for k, v in x.items()}
            
            with torch.no_grad():
                output = model(**x)

            loss = output['loss']
            logits = output['logits']            
            
            predictions = logits.cpu().detach().numpy().argmax(axis=-1)
            label = x['labels'].cpu().detach().numpy()
            batch_indices = batch_idx * self.test_loader.batch_size + np.arange(self.test_loader.batch_size)
            
            incorrect_indices = np.where(predictions != label)[0]
            for incorrect_index in incorrect_indices:
                miss_predict.append({
                    'index': batch_indices[incorrect_index],
                    'prediction': predictions[incorrect_index],
                    'label': label[incorrect_index]
            })
            
            # total_logits.append(logits.cpu().detach().numpy().argmax(axis=-1))
            # total_labels.append(x['labels'].cpu().detach().numpy())
            # indices.append(batch_idx * self.test_loader.batch_size + np.arange(self.test_loader.batch_size))
            
            total_logits.append(predictions)
            total_labels.append(label)
            indices.append(batch_indices)
                
            total_output['loss'].append(loss.cpu().item())
            
        total_logits = np.concatenate(total_logits, axis=0)
        total_labels = np.concatenate(total_labels)
        
        metric_output = self.calculate_metric(total_logits, total_labels)
        total_output['loss'] = np.mean(total_output['loss']).item()
        for k in metric_output:
            total_output[k] = metric_output[k]
        
        df = pd.DataFrame(total_output, index=[0])
        df2 = pd.DataFrame(miss_predict)
    #     df2 = pd.DataFrame({
    #     'index': index,
    #     'prediction': miss_predict,
    #     'label': label
    # })
        df.to_csv(os.path.join(self.outputs_dir, "test_output.csv"), index=False)
        df2.to_csv(os.path.join(self.outputs_dir, "test_index_output.csv"), index=False)

        return total_output