import os

################## GPU #####################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
############################################

import torch
import numpy as np
import random
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.text_utils import load_from_csv, split_data
from utils.data_utils import make_loader
from dataset import SMSDataset
from trainer import Trainer
import transformers as tf
from model import ModiModel
# from transformers import BertForSequenceClassification, BertTokenizerFast

def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

if __name__  == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    seed=args.seed
    random_seed(seed=seed)
    torch.cuda.init()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("-----CUDA DEVICES-----")
    print('Device:', args.device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    print(f"Model: epochs_{args.epochs}_seed_{seed}_lr_{args.lr}_batch_{args.batchsize}")
    print("-" * 22)
    
    id2label = {0:'ham',
                1:'smishing',
                2:'spam'}

    label2id = {'ham': 0,
                'smishing': 1,
                'spam': 2}
    
    
    # model_path = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
    model_path = 'bert-base-uncased'
    # model_path = 'roberta-base'
    # model_path = 'albert-base-v2'
    # bert_model = tf.AlbertForSequenceClassification.from_pretrained(model_path,
    #                                                                 num_labels=3,
    #                                                                 id2label=id2label,
    #                                                                 label2id=label2id)
    # bert_tokenizer = tf.AlbertTokenizerFast.from_pretrained(model_path)
    # bert_model = tf.BertForSequenceClassification.from_pretrained(model_path,
    #                                                             num_labels=3,
    #                                                             id2label=id2label,
    #                                                             label2id=label2id)
    # bert_tokenizer = tf.BertTokenizer.from_pretrained(model_path)
    bert_model = tf.BertForSequenceClassification.from_pretrained(model_path,
                                                                       num_labels=3,
                                                                       id2label=id2label,
                                                                       label2id=label2id)
    bert_tokenizer = tf.BertTokenizer.from_pretrained(model_path)
    # bert_model = ModiModel(model_path, 3)
    
    # bert_tokenizer = tf.AutoTokenizer.from_pretrained(model_path)
    
    # bert_model.albert.pooler.dropout = nn.Dropout(0.1)
    # bert_model.classifier.dropout = nn.Dropout(0.1)
    
    
    train_data = pd.read_csv(f"data/seed_{seed}/data_train_seed_{seed}.csv")
    valid_data = pd.read_csv(f"data/seed_{seed}/data_valid_seed_{seed}.csv")
    test_data = pd.read_csv(f"data/seed_{seed}/data_test_seed_{seed}.csv")
    
    
    train_set = SMSDataset(data=train_data,
                       tokenizer=bert_tokenizer,
                       max_length=bert_model.config.max_position_embeddings)
    
    valid_set = SMSDataset(data=valid_data,
                       tokenizer=bert_tokenizer,
                       max_length=bert_model.config.max_position_embeddings)
    
    test_set = SMSDataset(data=test_data,
                       tokenizer=bert_tokenizer,
                       max_length=bert_model.config.max_position_embeddings)
    
    train_loader = make_loader(dataset=train_set, batch_size=args.batchsize,seed=seed)
    valid_loader = make_loader(dataset=valid_set, batch_size=args.batchsize,seed=seed)
    test_loader = make_loader(dataset=test_set, batch_size=args.batchsize,seed=seed)
    
    output_dir = f"outputs/epochs_{args.epochs}_seed_{seed}_lr_{args.lr}_batch_{args.batchsize}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # bert_model = DistributedDataParallel(bert_model)
        
    trainer = Trainer(train_loader=train_loader,
                      valid_loader=valid_loader,
                      test_loader=test_loader,
                      learning_rate=args.lr,
                      num_epochs=args.epochs,
                      device=args.device,
                      outputs_dir=output_dir,
                    #   warmup_steps=2,
                      logger_name=f"logs/epochs_{args.epochs}_seed_{seed}_lr_{args.lr}_batch_{args.batchsize}")
    
    trainer.fit(model=bert_model)