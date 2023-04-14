Team Spirit
## Overview

## Data
As BERT model has a max 512 length limit of tokens, we split each document into small chunks with size 512.
Also, the BERT model was built on the 2 different size datasets, one is 2000 * 512 tokens per author, anothor is 1000 * 512 per author.
The 80%, 10%, 10% dataset was split into train, test and validation dataset.

## Requirements
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
1. Please check the tran, test and validation datasets are all in the same directory of the transformer.py
2. This is the default values setting:
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 24
    VALID_BATCH_SIZE = 24
    EPOCHS = 6 # 20
    LEARNING_RATE = 2e-05
    * If you want to change the values, please modify them in the transformer.py.
3. The transformer.py will run 4 times with 4 different techniques: 
        BERT default pooling max pooling, concatenate pooling and mean pooling
    * If you want to change the techniques, please modify the patterns list (line 311)

## Run the code

```
   $ python3 transformer.py
```

## Result
The result of the measurement of the 4 different techniques will be save as transformer_stats.csv in the same directory as the transformer.py.
