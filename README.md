# PILD
Project for Personal Information Leakage Detection in Conversations

### Request
Pytorch, [trec_eval](https://trec.nist.gov/trec_eval/) 


### Data Pre-processing
mkdir model output

> python preprocess_rep.py -lm bert -input ./dataset/persona_linking_test.json -output ./dataset/persona_linking_test.bert
> python preprocess_rep.py -lm bert -input ./dataset/persona_linking_train.json -output ./dataset/persona_linking_train.bert

### Model Training
> python train.py -epochs 200 -save_model ./model/bert_ -method att_sparse -alpha 0.4 \
> -train_dataset ./dataset/persona_linking_train.bert -dev_dataset ./dataset/persona_linking_dev.bert

> python train.py -epochs 200 -save_model ./model/bert_ -method att_sharp -alpha 0.4 -gamma 6.0 \
> -train_dataset ./dataset/persona_linking_train.bert -dev_dataset ./dataset/persona_linking_dev.bert


### test SPARSE & SHARP model (Note: install TREC_EVAL first)
> python test.py -model ./model/bert_att_sparse_0.01_0.4_E200* -test_dataset ./dataset/persona_linking_test.bert -test_result ./output/bert_att_sparse_0.01_0.4_E200.result
> ~/Desktop/trec_eval-9.0.7/trec_eval output/persona_linking_test.txt output/bert_att_sparse_0.01_0.4_E200.result -m map -m P.1,2,3,5 -m Rprec -m ndcg


> python test.py -model ./model/bert_att_sharp_0.01_0.4_6.0_E200* -test_dataset ./dataset/persona_linking_test.bert -test_result ./output/bert_att_sharp_0.01_0.4_6.0_E200.result
> ~/Desktop/trec_eval-9.0.7/trec_eval output/persona_linking_test.txt output/bert_att_sharp_0.01_0.4_6.0_E200.result -m map -m P.1,2,3,5 -m Rprec -m ndcg
