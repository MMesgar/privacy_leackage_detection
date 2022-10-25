
import torch
from transformers import *
import opts, util
import numpy as np
from tqdm import tqdm
import argparse, sys

import nn_model

parser = argparse.ArgumentParser(
    description='train.py')
opts.test_opts(parser)
opt = parser.parse_args()

np.random.seed(opt.seed)

def test():
    test_dataset = torch.load(opt.test_dataset,map_location=torch.device('cpu'))
    
    if opt.model is not None:
        model = torch.load(opt.model)  
        print("load pretrained model")
    else:
        model = None

    if opt.cuda:
        test_utterance = [u.cuda() for u in test_dataset['utterance']]
        test_persona = [u.cuda() for u in test_dataset['persona']]

        if model:
            model = model.cuda()
    else:
        test_utterance = test_dataset['utterance']
        test_persona = test_dataset['persona']

    assert len(test_persona) == len(test_utterance)

    with open(opt.test_result, 'w') as result_f:
        
        for i in range(len(test_utterance)):
            if opt.method == 'bert' or opt.method == 'rand':
                utt_rep, per_rep = test_utterance[i], test_persona[i]
                utt_rep=torch.squeeze(utt_rep)
                per_rep=torch.squeeze(per_rep)
                
            else:
                utt_rep, per_rep = model.linear(torch.squeeze(test_utterance[i])), model.linear(torch.squeeze(test_persona[i])) #sparsemax, sharpmax, softmax
                
                
            

            sim = nn_model.pairwise_cosine(utt_rep, per_rep).data.cpu().numpy()
            u_num, p_num = sim.shape
            for u_idx in range(u_num):
                for p_idx in range(p_num):
                    if opt.method == 'rand':
                        result_f.write("d{}_u{} Q0 p{} 0 {} STANDARD\n".format(i, u_idx, p_idx+1, np.random.rand()))
                    else:
                        result_f.write("d{}_u{} Q0 p{} 0 {} STANDARD\n".format(i,u_idx, p_idx+1,  sim[u_idx, p_idx]))
    print('finish testing')

if __name__ == '__main__':
    test()