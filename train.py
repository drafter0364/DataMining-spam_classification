# %%
import pandas as pd

# %%
train = pd.read_csv('./data/train.csv', sep=',', names=['ID', 'Label','Email'],skiprows=1) 
train['Label'] = train['Label'].apply(lambda x: 1 if x == 'spam' else 0) 

test = pd.read_csv('./data/test_noLabel.csv', sep=',', names=['ID', 'Email'],skiprows=1)

# %%
trainData = list(train['Email'])
testData = list(test['Email'])
labels = list(train['Label'])

# %%
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=True)

# %%
input_ids0 = []
attention_masks0 = []

for sent in trainData:
    encoded_dict0 = tokenizer.encode_plus(
                        sent,     
                        add_special_tokens = True, 
                        max_length = 64,          
                        pad_to_max_length = True,
                        return_attention_mask = True,  
                        return_tensors = 'pt',     
                   )
    
    input_ids0.append(encoded_dict0['input_ids'])
    attention_masks0.append(encoded_dict0['attention_mask'])
    

# %%
import torch

input_ids0 = torch.cat(input_ids0, dim=0)
attention_masks0 = torch.cat(attention_masks0, dim=0)
labels = torch.tensor(labels)


# %%
from torch.utils.data import TensorDataset, random_split
dataset = TensorDataset(input_ids0, attention_masks0, labels)

# %%
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
batch_size = 64
train_dataloader = DataLoader(
            dataset,
            sampler = RandomSampler(dataset),
            batch_size = batch_size
        )

# %%
if torch.cuda.is_available():      
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# %%
from transformers import BertForSequenceClassification, AdamW, BertConfig

model = BertForSequenceClassification.from_pretrained(
    "bert-base-cased",
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False, 
)
model.to(device)

# %%
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, 
                  eps = 1e-8
                )

# %%
from transformers import get_linear_schedule_with_warmup

epochs = 10
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

# %%
import numpy as np
import time
import datetime

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# %%
import random
import numpy as np

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
training_stats = []

# 每一轮训练：
for epoch_i in range(0, epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):

        if step % 40 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        

        output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = output.loss
        logits = output.logits
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_train_loss / len(train_dataloader)            

    print("")
    print("  Average training loss: {0:.5f}".format(avg_train_loss))

# %%
torch.save(model.state_dict(), "./model_01.pth")

# %%



