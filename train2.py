#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install -q transformers')
#get_ipython().system('pip install -q datasets jiwer')


# In[1]:


import pandas as pd

df = pd.read_table('./data/train/caption.txt', header=None) #fwf
df.rename(columns={0: "file_name", 1: "text"}, inplace=True)

df['file_name']= df['file_name'].apply(lambda x: x+'.jpg')
df = df.dropna()
df


# In[3]:



df2 = pd.read_table('./data/2014/caption.txt', header=None) #fwf
df2.rename(columns={0: "file_name", 1: "text"}, inplace=True)

df2['file_name']= df2['file_name'].apply(lambda x: x+'.jpg')
df2 = df2.dropna()
#fliter = (df2["file_name"] == "505_em_51.bmp")
df2


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

train_df = df
test_df = df2
#train_df, test_df = train_test_split(df, test_size=0.2) #shuffle =True
train_df = shuffle(train_df)
#test_df = shuffle(df2)

# we reset the indices to start from zero
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

train_df


# In[5]:


train_df.head(10)


# In[6]:


import torch
from torch.utils.data import Dataset
from PIL import Image

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=490):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


# In[7]:


from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten") #"microsoft/trocr-base-handwritten"
train_dataset = IAMDataset(root_dir='./data/train/',
                           df=train_df,
                           processor=processor)
eval_dataset = IAMDataset(root_dir='./data/2014/', #'./data2/2014/'
                           df=test_df,
                           processor=processor)


# In[8]:


#from torch.utils.data import DataLoader

#train_loader = DataLoader(train_dataset, batch_size = 8, shuffle = True, num_workers = 4)
#val_loader = DataLoader(eval_dataset, batch_size = 8, shuffle = True, num_workers = 4)


# In[8]:


print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))


# In[9]:


encoding = train_dataset[0]
for k,v in encoding.items():
  print(k, v.shape)


# In[10]:


image = Image.open(train_dataset.root_dir + train_df['file_name'][0]).convert("RGB")
image


# In[11]:


labels = encoding['labels']
labels[labels == -100] = processor.tokenizer.pad_token_id
label_str = processor.decode(labels, skip_special_tokens=True)
print(label_str)


# In[12]:


from transformers import VisionEncoderDecoderModel

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-stage1") #microsoft/trocr-base-stage1


# In[13]:


# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 490  
model.config.early_stopping = True
#model.config.no_repeat_ngram_size = 2
#model.config.length_penalty = 2.0
model.config.num_beams = 10

#原本只有 model.config.early_stopping = True model.config.num_beams = 10


# In[14]:


from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=8, #origin 8
    per_device_eval_batch_size=1, #origin 8
    fp16=True, 
    output_dir="./checkpoint_eval_2014_small_stage1_new_image/",
    logging_steps=2,
    save_steps=1000,
    eval_steps=500,
    num_train_epochs = 100,
)


# In[ ]:


#from datasets import load_metric

#cer_metric = load_metric("cer")
#cer_metric = load_metric("accuracy")


# In[ ]:


'''
import numpy as np

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}
'''
'''
def compute_metrics(pred):
    predictions= pred.predictions
    labels = pred.label_ids
    predictions = np.argmax(predictions, axis=1)
    return cer_metric.compute(predictions=predictions, references=labels)
'''


# In[15]:


from transformers import default_data_collator

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    #compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)
trainer.train() #"checkpoint-9000"


# Inference

# In[30]:


#tensorboard --logdir checkpoint_eval_2014/


# In[14]:


#predictions = trainer.predict(eval_dataset)
#image = Image.open(eval_dataset.root_dir + test_df['file_name'][3]).convert("RGB")
#image


# In[17]:


#encoding = eval_dataset[3]
#for k,v in encoding.items():
#  print(k, v.shape)


# In[18]:


#labels = encoding['labels']
#labels[labels == -100] = processor.tokenizer.pad_token_id
#label_str = processor.decode(labels, skip_special_tokens=True)
#print(label_str)


# In[ ]:


#processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#model = VisionEncoderDecoderModel.from_pretrained("./checkpoint-9000")


# In[ ]:


#def ocr_image(src_img):
#  pixel_values = processor(images=src_img, return_tensors="pt").pixel_values
#  generated_ids = model.generate(pixel_values)
#  return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


# In[ ]:


#ocr_image(image)

