# %%
import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from transformers import BertForSequenceClassification, AdamW, BertConfig

# %%
test = pd.read_csv('./data/test_noLabel.csv', sep=',', names=['ID', 'Email'],skiprows=1)
testData = list(test['Email'])

# %%
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# %%
def clearmemory(model,checkpoint,tokenizer):
    torch.cuda.empty_cache()
    del model
    del checkpoint
    del tokenizer

# %%
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
checkpoint = torch.load('./model.pth')

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2, 
    output_attentions = False,
    output_hidden_states = False, 
)
model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

# %%
label0 = []
probabilities0 = []
for idx, sent in enumerate(testData):
    encoded_dict = tokenizer.encode_plus(sent, add_special_tokens = True, max_length = 64, padding='longest',return_attention_mask = True,return_tensors = 'pt')
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)
    outputs = model(input_ids, attention_mask)
    logits = outputs.logits
    label0.append(torch.argmax(logits, dim=1).item())
    probabilities0.append(torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()[0])

# %%
# clearmemory(model,checkpoint,tokenizer)

# %%
'''
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
checkpoint = torch.load('./model.pth')

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2, 
    output_attentions = False, 
    output_hidden_states = False, 
)
model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

# %%
label1 = []
probabilities1 = []
for idx, sent in enumerate(testData):
    encoded_dict = tokenizer.encode_plus(sent, add_special_tokens = True, max_length = 64, padding='longest',return_attention_mask = True,return_tensors = 'pt')
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)
    outputs = model(input_ids, attention_mask)
    logits = outputs.logits
    label1.append(torch.argmax(logits, dim=1).item())
    probabilities1.append(torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()[0])

# %%
clearmemory(model,checkpoint,tokenizer)

# %%
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
checkpoint = torch.load('./model.pth')

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2,
    output_attentions = False, 
    output_hidden_states = False, 
)
model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

# %%
label2 = []
probabilities2 = []
for idx, sent in enumerate(testData):
    encoded_dict = tokenizer.encode_plus(sent, add_special_tokens = True, max_length = 64, padding='longest',return_attention_mask = True,return_tensors = 'pt')
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)
    outputs = model(input_ids, attention_mask)
    logits = outputs.logits
    label2.append(torch.argmax(logits, dim=1).item())
    probabilities2.append(torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()[0])

# %%
clearmemory(model,checkpoint,tokenizer)
'''
# %%
#assert(len(label0)==len(label1))
#assert(len(label0)==len(label2))
label_forward = []
# valid_forward = []
for i in range(len(label0)):
    # label = 0 if label0[i] + label1[i] + label2[i] < 2 else 1
    label = label0[i]
    label_forward.append(label)
    # p0, p1, p2 = probabilities0[i],probabilities1[i],probabilities2[i]
    # valid = 0 if (0.002 < p0 < 0.998) or (0.002 < p1< 0.998) or (0.002 < p2 < 0.998) else 1
    # valid_forward.append(valid)

# %%
'''
label_backward = label_forward[:]

# %%
spam_words = []
ham_words = []
key_words = []
with open('/kaggle/input/forwords/spam_words.txt', 'r') as f:
    spam_words = [line.strip() for line in f.readlines()]
with open('/kaggle/input/forwords/ham_words.txt', 'r') as f:
    ham_words = [line.strip() for line in f.readlines()]
with open('/kaggle/input/forwords/key_words.txt', 'r') as f:
    key_words = [line.strip() for line in f.readlines()]

        

# %%
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
checkpoint = torch.load('/kaggle/input/smallbert/smallbert.pth')

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2,
    output_attentions = False, 
    output_hidden_states = False, 
)
model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

# %%
for i in range(len(label0)):
    if valid_forward[i] == 0:
        for word in key_words:
            if word in testData[i]:
                encoded_dict = tokenizer.encode_plus(testData[i], add_special_tokens = True, max_length = 64, padding='longest',return_attention_mask = True,return_tensors = 'pt')
                input_ids = encoded_dict['input_ids'].to(device)
                attention_mask = encoded_dict['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask)
                logits = outputs.logits
                label_backward[i] = torch.argmax(logits, dim=1).item()
                
        for word in spam_words:
            if word in testData[i]:
                label_backward[i] = 1
        for word in ham_words:
            if word in testData[i]:
                label_backward[i] = 0
'''
# %%
predictions_df = pd.DataFrame(columns=['ID', 'Label'])
for i in range(len(label0)):
    # predictions_df.loc[i] = [test["ID"][i],"ham" if label_backward[i] == 0 else "spam"]
    predictions_df.loc[i] = [test["ID"][i],"ham" if label_forward[i] == 0 else "spam"]
predictions_df.to_csv('output.csv', index=False)


