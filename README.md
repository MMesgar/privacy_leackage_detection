## PLD
Project and Dataset for Privacy Leakage Detection in Conversations.

## Environment

torch >= 1.7.1
transformers==4.17.0
scipy
datasets
pandas
scikit-learn
prettytable
gradio
setuptools

### Dataset Preparation

Our dataset is from NLI task. It is a csv file. The first column is anchor sample, the secound column is positive sample, if using hard negative, the third column is hard negative sample. The anchor and postive samples are sentence pairs with entailment labels. If one anchor has several entailment sentences, we only select one to avoid false negatives.

All data files used are put in the dataset folder.

### Model Training

You can train NLICL model by running:

```
python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/DNLI_all_pairs..csv \
    --output_dir result/DNLI_all_pairs \
    --num_train_epochs 1 \
    --per_device_train_batch_size 128 \
    --learning_rate 5e-5 \
    --max_seq_length 25 \
    --pooler_type cls \
    --temp 0.05 \
    --do_train \
    
python pild_to_huggingface.py --result/DNLI_all_pairs    
```
You can set arguments above accourding to your need. After converting the NLICL checkpoint to Huggingface style, you can use the model fot esting

### Model Testing 
First, make sure you setup proper path of input(path to the file 'persona_linking_test.json'), pld_model and output_result in test_pld.py. Then run test_pld.py to get result of cosine similarity of each sentence pairs in PERSONA-LEAKAGE test dataset.

[trec_eval](https://trec.nist.gov/trec_eval/) is required.

If the directory to output result is 'output/DNLI_all_pairs.result', path to trec_eval is '/Desktop/trec_eval-9.0.7/trec_eval'. Run the following to print 5 evaluation metrics (P@1, P@2, Rprec, MAP, NDCG):

```
/Desktop/trec_eval-9.0.7/trec_eval output/persona_linking_test_utterance_based.txt output/DNLI_all_pairs.result -m map -m P.1,2 -m Rprec -m ndcg
```



