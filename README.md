# Team Spirit

For Team Spirits project we tackle author identification using 2 models, one based on SVM and one based on BERT

## Datasets Used
We used three datasets:
1. The top 30 authors with 50 documents each with a 60, 30, 10% split on the train, test and validation set respectively, https://unsw-my.sharepoint.com/:f:/g/personal/z5316653_ad_unsw_edu_au/Eol-3sqFPrhLsBt_ifuuF7kBv_XXpSkD8F1QB3HoI3clsA?e=J72fXn
2. The same 30 authors with 2000*512 tokens each with a 80, 10, 10% split on the train, test and validation set respectively, https://unsw-my.sharepoint.com/:u:/g/personal/z5408671_ad_unsw_edu_au/ERc49oO5kvVKieNhctoLLAEBqVMFZr4sUQ_MiawyKLInGA?e=20Pvdt
3. The same 30 authors with 1000*512 tokens each with a 80, 10, 10% split on the train, test and validation set respectively, https://unsw-my.sharepoint.com/:u:/g/personal/z5408671_ad_unsw_edu_au/ER0ph37G5dNOr7mX3WnOFKMBXhu_FFc7e-6eijhxJPcuhA?e=9fmPOs


## SVM Overview
+ Character n-grams as features
+ Used Stacked denoising autoencoders to enrich features
+ Encoded features fed into SVM classifier

## SVM Requirements
+ python==3.10.8
+ numpy==1.23.5
+ pandas==2.0.0
+ torch==1.11.0
+ scikit-learn==1.2.2
+ matplotlib==3.7.1
+ keras==2.12.0
+ tensorflow=2.12.0

## SVM Data
+ Two SVM model were built using both datasets, dataset 1. was used for the sole purpose of training the SVM whilst dataset 2. was used to compare the BERT and SVM models.
+ The SVM model built on dataset 1. is written in svm_original.ipynb and the SVM model built on dataset 2. is written in svm_bert_data.ipynb.
+ The SVM model built on dataset 2. uses a GPU-accelerated SVM due to the large row size of the dataset and the RAPID cuml.svm package that implements LinearSVC was used for the SVM classifier. The installation of this depends on the architecture of the operating system and the installation guide can be found here https://docs.rapids.ai/install?_gl=1*1wfzw4y*_ga*MTUyMjg2OTQwLjE2ODEzNTg3Mzc.*_ga_RKXFW6CM42*MTY4MTM4MjU4Ni4zLjAuMTY4MTM4MjU4Ni4wLjAuMA..

## BERT Overview
+ We use the Bert-base uncased model​
+ Each paragraph generates word embeddings and word position embeddings.​
+ Feed into 12 transformer layers, every layer learns connections between words by Attention Mechanism.​
+ Connect a pooling layer to aggregate information.​
+ We compare with BERT default pooling, max pooling, concatenate pooling, mean pooling.
+ Apply a linear classifier layer in the last layer.​
+ Fine-tuning the weights is applied only on the last 4 transformer layers, the pooling layer and the classifier layer which can significantly save training time.

## BERT Data
+ As BERT model has a max 512 length limit of tokens, we split each document into small chunks with size 512.
+ Also, the BERT model was built datasets 2. and 3., one is 2000 * 512 tokens per author, anothor is 1000 * 512 per author.
+ The 80%, 10%, 10% dataset was split into train, test and validation dataset.

## BERT Requirements
+ python==3.10.8
+ numpy==1.23.5
+ pandas==2.0.0
+ transformers==4.27.4
+ torch==1.11.0
+ torch-scatter==2.0.2
+ scikit-learn==1.2.2
+ tqdm==4.65.0
+ matplotlib==3.7.1

## Before running
1. <b> Please check the tran, test and validation datasets are all in the same directory of the `transformer.py`. </b>
2. This is the default values setting:

    ```
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 24
    VALID_BATCH_SIZE = 24
    EPOCHS = 6 # 20
    LEARNING_RATE = 2e-05
    ```
    * If you want to change the values, please modify them in the `transformer.py`.
3. The `transformer.py` will run 4 times with 4 different techniques: 
        BERT default pooling max pooling, concatenate pooling and mean pooling
    * If you want to change the techniques, please modify the patterns list (line 311, `transformer.py`).

## Run the code

```
   $ python3 transformer.py
```

## Result
The result of the measurement of the 4 different techniques will be saved as `transformer_stats.csv` in the same directory as the transformer.py.
