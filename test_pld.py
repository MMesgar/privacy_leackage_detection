#test utterance based

import json
from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

def extract_all_samples(dataset):
    samples = []
    utterances = []
    personas = []
    for d in dataset:
        ua = [u['A'][0] for u in d['dialogue']]
        ub = [u['B'][0] for u in d['dialogue']]
        pa = [p for _, p in d['pa'].items() if not pd.isnull(p) and len(p) > 0]  # a=" " len(a)=1
        pb = [p for _, p in d['pb'].items() if not pd.isnull(p) and len(p) > 0]
        samples.append((ua, pa))
        samples.append((ub, pb))

        utterances += ua + ub
        personas += pa + pb

    return samples, utterances, personas

def pairwise_cosine(m1, m2=None, eps=1e-6):
    if m2 is None:
        m2 = m1
    w1 = m1.norm(p=2, dim=1, keepdim=True)
    w2 = m2.norm(p=2, dim=1, keepdim=True)

    return torch.mm(m1, m2.t()) / (w1 * w2.t()).clamp(eps)

input='persona_linking_test.json'
pld_model='...'    #path to model needs to be evaluted
output_result='...'  #path to output result


tokenizer = AutoTokenizer.from_pretrained(pild_model)
model = AutoModel.from_pretrained(pild_model)
model.to(device)

with open(input, 'r') as f:
    # data preparation
    train_dataset = json.load(f)
    samples_text, utterances, personas = extract_all_samples(train_dataset)
  

    print('dialogues[{}], utterances[{}], personas[{}]'.format(
        len(samples_text) / 2, len(utterances), len(personas)))

u_d = []
p_d = []

for i in tqdm(range(len(samples_text))): 
    u_s = samples_text[i][0]
    p_s = samples_text[i][1]
   
    u_tmp = torch.stack([model(torch.tensor([tokenizer.encode(u_t)]).to(device)).pooler_output for u_t in u_s], dim = 0)
    p_tmp = torch.stack([model(torch.tensor([tokenizer.encode(p_t)]).to(device)).pooler_output for p_t in p_s], dim = 0)
   
    
    u_d.append(u_tmp.data) 
    p_d.append(p_tmp.data)


with open(output_result, 'w') as result_f:
        
    for i in range(len(u_d)):
        utt_rep, per_rep = u_d[i], p_d[i]
        utt_rep=torch.squeeze(utt_rep)
        per_rep=torch.squeeze(per_rep)
                                
        sim = pairwise_cosine(utt_rep, per_rep).data.cpu().numpy()
        u_num, p_num = sim.shape
        for u_idx in range(u_num):
            for p_idx in range(p_num):
                result_f.write("d{}_u{} Q0 p{} 0 {} STANDARD\n".format(i,u_idx, p_idx+1,  sim[u_idx, p_idx]))
print('finish testing')
 
