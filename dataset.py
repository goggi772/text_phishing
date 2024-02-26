import torch
import random
import numpy as np
from typing import Union, List, Dict
import nlpaug.augmenter.word as naw
from torch.utils.data import Dataset, DataLoader

class SMSDataset(Dataset):
    def __init__(self,
                 data: Dict[str, List],
                 tokenizer = None,
                 label_encoder=None,
                 max_length:int=512
                 ):

        self.data = data['TEXT']
        self.label = data['LABEL_ID']
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_encoder = label_encoder
                    
    def __len__(self):
        return len(self.label)
    
    def encode_sample(self, sample: Union[str, List[str]]):
        return self.tokenizer(sample,
                              padding=True,
                              truncation=True,
                              max_length = self.max_length,
                              return_tensors='pt')
    
    def __getitem__(self, idx):
        
        # selected_text = self.data[idx] if random.random() < 0.5 else augmented_text  #증강한 텍스트를 절반만 가져옴
        
        # elements = {'data': self.augment_text(self.data[idx]),
                    # 'label': self.label[idx]}
        elements = {'data': self.data[idx],
                    'label': self.label[idx]}

        return elements
    
    def augment_text(self, text):
    # 대체 어구를 사용하여 텍스트를 증강
        text = str(text)
        aug = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased',
            action="substitute"
        )
        augmented_text = aug.augment(text)
        return augmented_text

    def collate_fn(self, samples:Dict[str, List]):
        # augmented_datas = [ s['augmented_data'] for s in samples]
        datas = [ s['data'] for s in samples]
        labels = [ s['label'] for s in samples]
        
        # augmented_datas = self.encode_sample(augmented_datas)
        datas = self.encode_sample(datas)
        labels = torch.tensor(labels)

        elements = { k:v for k, v in datas.items()}
        # elements.update({k + "_augmented": v for k, v in augmented_datas.items()})
        elements['labels'] = labels

        return elements
        
    def random_seed(self, seed:int = 42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
    def id2label(self, label:Union[List[int], int]):
        return self.label_encoder.inverse_transform(label)