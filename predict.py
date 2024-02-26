import os

################## GPU #####################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
############################################


import torch
import numpy as np
import random
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.text_utils import load_from_csv, split_data
from utils.data_utils import make_loader
from dataset import SMSDataset
from trainer import Trainer
import transformers as tf
# from transformers import BertForSequenceClassification, BertTokenizerFast

def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

if __name__  == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    
    seed=args.seed
    random_seed(seed=seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("-----CUDA DEVICES-----")
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    print("-" * 22)
    
    id2label = {0:'ham',
                1:'smishing',
                2:'spam'}

    label2id = {'ham': 0,
                'smishing': 1,
                'spam': 2}
    
    # model_path = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
    model_path = 'bert-base-uncased'
    # model_path = 'albert-base-v2'
    output_dir = f"outputs/epochs_{args.epochs}_seed_{seed}_lr_{args.lr}_batch_{args.batchsize}"
    
    
    # bert_model = tf.AlbertForSequenceClassification.from_pretrained(os.path.join(output_dir, 'model_best_valid_accuracy'))
    # bert_tokenizer = tf.AlbertTokenizerFast.from_pretrained(model_path)
    # bert_model = tf.BertForSequenceClassification.from_pretrained(os.path.join(output_dir, 'model_best_valid_accuracy'))
    # bert_tokenizer = tf.BertTokenizer.from_pretrained(model_path)
    bert_model = tf.AutoModelForSequenceClassification.from_pretrained(os.path.join(output_dir, 'model_best_valid_accuracy'))
    bert_tokenizer = tf.AutoTokenizer.from_pretrained(model_path)
    
    test_data = pd.read_csv(f"data/seed_{seed}/data_test_seed_{seed}.csv")
    test_set = SMSDataset(data=test_data,
                       tokenizer=bert_tokenizer,
                       max_length=bert_model.config.max_position_embeddings)
    test_loader = make_loader(dataset=test_set, batch_size=args.batchsize, seed=seed, istest=True)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    trainer = Trainer(test_loader=test_loader,
                      device=device,
                      outputs_dir=output_dir)
    
    test_output = trainer.test(model=bert_model)
    print(f"test_output:{test_output}")