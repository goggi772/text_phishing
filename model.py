import torch
import transformers as tf
import torch.nn as nn

class ModiModel(nn.Module):
    def __init__(self, bert_model, num_class):
        super(ModiModel, self).__init__()
        
        id2label = {0:'ham',
                1:'smishing',
                2:'spam'}

        label2id = {'ham': 0,
                'smishing': 1,
                'spam': 2}
    
        self.config = tf.AutoConfig.from_pretrained(bert_model)
        # self.config.id2label = id2label
        # self.config.label2id = label2id

        
        # self.bert = tf.BertModel.from_pretrained(bert_model,
        #                                         config=self.config)
        self.bert = tf.AutoModel.from_pretrained(bert_model,
                                                                     config=self.config)
        self.tokenizer = tf.AutoTokenizer.from_pretrained(bert_model)
        
        # self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.dense = nn.Linear(self.config.hidden_size, num_class)
        # nn.init.trunc_normal_(self.dense.weight.data)
        
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        outputs = self.bert(input_ids)
        # outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        # pooler = self.bert(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)
        # pooled_output = self.dropout(outputs[1])
        # logits=self.dense(pooled_output)
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]
        logits = self.dense(cls_hidden_state)
        # output = self.linear_layer(cls_hidden_state)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss
        else:
            return logits
        

