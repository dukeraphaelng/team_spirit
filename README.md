# Author Identification on the English Project Guternberg Corpus 

COMP9417 - Machine Learning & Data Mining (Group Project) & Berrijam Jam Submission 

  

**Group**: Team Spirit 

  

**Authors**: Duke Nguyen (z5398432), Lavender Kong (z5271686), Jaeff Hong (z5316653), Weixian Qian (z5408671).  

  

## Installation 

1. Download the dataset 

+ Download the dataset at the following links: 

  + 1. The top 30 authors with 50 documents each with a 60, 30, 10% split on the [train, test and validation set respectively](https://unsw-my.sharepoint.com/:f:/g/personal/z5316653_ad_unsw_edu_au/Eol-3sqFPrhLsBt_ifuuF7kBE-4U99KdZj35aN497nrn4Q?e=0fBZ9X)

  + 2. The same 30 authors with 2000*512 tokens each with a 80, 10, 10% split on the [train, test and validation set respectively](https://unsw-my.sharepoint.com/:u:/g/personal/z5408671_ad_unsw_edu_au/ERc49oO5kvVKieNhctoLLAEBqVMFZr4sUQ_MiawyKLInGA?e=dwDJpl)

  + 3. The same 30 authors with 1000*512 tokens each with a 80, 10, 10% split on the [train, test and validation set respectively](https://unsw-my.sharepoint.com/:u:/g/personal/z5408671_ad_unsw_edu_au/ER0ph37G5dNOr7mX3WnOFKMBV7zzPL169hXTGkrNCl1euw?e=EMma4v)

Alternatively, you can generate the dataset by following these steps 

+ Download Standardized Project Gutenberg Corpus (version SPGC-2018-07-18) with the following three files: 

  + SPGC-counts-2018-07-18.zip 

  + SPGC-metadata-2018-07-18.csv 

  + SPGC-tokens-2018-07-18.zip 

+ Run ` python3 sampling_data/preprocessing.py` to obtain file no. 1 

+ Change n=2000 at line 479 to n=1000 , and run ` python3 sampling_data/preprocessing.py` to obtain file no. 2 and file no. 3 

 

2. EDA 

- You can run the `EDA.ipynb` notebook to look at the data analysis 


3. Requirements

**SVM**

+ python==3.10.8 

+ numpy==1.23.5 

+ pandas==2.0.0 

+ torch==1.11.0 

+ scikit-learn==1.2.2 

+ matplotlib==3.7.1 

+ keras==2.12.0 

+ tensorflow=2.12.0 

**BERT**

+ python==3.10.8 

+ numpy==1.23.5 

+ pandas==2.0.0 

+ transformers==4.27.4 

+ torch==1.11.0 

+ torch-scatter==2.0.2 

+ scikit-learn==1.2.2 

+ tqdm==4.65.0 

+ matplotlib==3.7.1 

4. Before running 

+ <b> Please check the tran, test and validation datasets are all in the same directory of the `transformer.ipynb`. </b> 
+ <b> Please check the tran, test and validation datasets are all in the same directory of the `svm_bert_data.ipynb`. </b> 

+ This is the default values setting for BERT: 

    ``` 

    MAX_LEN = 512 

    TRAIN_BATCH_SIZE = 24 

    VALID_BATCH_SIZE = 24 

    EPOCHS = 6 # 20 

    LEARNING_RATE = 2e-05 

    ``` 

    * If you want to change the values, please modify them in the `transformer.py`. 

+. The `transformer.py` will run 4 times with 4 different techniques:  

        BERT default pooling max pooling, concatenate pooling and mean pooling 

    * If you want to change the techniques, please modify the patterns list (line 311, `transformer.py`). 

5. Run the code 

+ Run `svm_bert_data.ipynb` then run `transformer.ipynb`
  
+ The result of the SVM, and BERT will be saved as `svm_stats.csv`, and `transformer_stats.csv` respectively in the same directory. 

## Introduction 

The predictive problem will be ‘Multiple class classification’. Specifically, it is ‘author identification’, this is different from, but related to the task of ‘authorship verification’. ‘Authorship Identification’ is the task of identifying the author of a text, given a set of candidate authors (Mohen 2016). In the machine learning domain, it can be classified as a ‘multi-class single-label text classification task’ where the text or features extracted from the text are features, and the author is the class label (Mohen 2016). Authorship identification can be used to identify the works of authors writing under pseudonyms. Historically, non-machine learning methods have been used to identify the authors of the Federalist Papers, for this purpose. It can also be used in textual criticism to identify which work is genuine, this is relevant in the research regarding the Analects by Confucius. In addition, it is also useful in a commercial setting to identify the authorship of quotes from famous people. Related applications in information security and forensics include plagiarism detection, suicide note authorship identification, etc. (Romanov et al. 2021).   

  

Previous works in this area, for example, include Mohsen et al. (2016), Romanov et al. (2020), Sarwar et al. (2018). Until 2016, this problem has usually been tackled using classical supervised learning models. Afterwards, deep learning models have been used. However, implementations of SOTA models like pre-trained BERT or BERT derivatives have been scarce or non-existent. BERT has been used similarly on ‘authorship verification’ (Manolache et al. 2021) (Kestemont et al. 2020), however.  

  

Therefore, we will be using SVM with SDAE, and BERT on the English SPGC (Standardized Project Gutenberg Corpus) (Gerlach and Fontclos 2020 (GNU General Public License). 

  

## Literature Review 

Previous works done in author identification mainly tackle two issues, the extraction of features of texts to represent writing styles of different authors and methods to correctly predict the author, given text.  

One such work [Mohsen et al. (2016)] that went with a deep learning approach, utilized character n-grams in conjunction with Stacked Denoising AutoEncoders (SDAE) for feature extraction on a corpus consisting 50 authors where each author had written 100 documents from Reuters Corpus Volume 1 (RCV1). Additionally, two feature selection techniques were compared, namely frequency based and Chi Square which were used to retain features with high information for a given category and to weed out redundant features. More specifically, unstructured text input was first converted to character n-grams using feature extraction and passed through to feature selection. The SDAE was then fed with the produced variable size n-gram feature set and the extracted features were used as input for a support vector machine (SVM) classifier. The result was a model that could outperform previous state-of-the-art author identification techniques when using the same corpus with a classification accuracy up to 95.12% whilst previous studies could only reach an accuracy of 80%. The comparisons showed that for a high feature space, the Chi Square based feature selection outperformed frequency-based feature selection and the reverse was true for a lower dimension space. Further, it showed the features extracted using SDAE outperformed previous approaches that simply only used fixed-length character n-grams, variable-length character n-grams and frequent words as features. Outlining the specifics of how data was split, 60% was used on the training set, 10% on the validation set and 30% on the test set. In addition, the SVM classifier was used using 10-fold cross validation. 

Another work [Romanov et al. (2020)] compared usage of Support Vector Machine (SVM) and modern classification methods based on deep Neural Networks (NNs) architectures, namely Long Short-Term Memory (LSTM), Convolutional neural networks (CNN) with attention and transformers. The corpus was collected from the Moshkov library which included 2086 texts written by 500 Russian authors. Obtained results from comparing between datasets of 2, 5, 10 and 50 candidate authors showed that the approach based on SVM demonstrated superior accuracy to modern deep NNs architectures, having an accuracy by more than 10% on average. 10-fold cross validation was used as a procedure for evaluating the effectiveness of all models and texts were divided into fragments ranging from 1000 to 100,000 characters (~200-20,000 words) where 3 training examples were used for each author, as well as one for testing. 

However, work presented in [Sarwar et al. (2018)] proposed a model based on a probabilistic k-nearest neighbor (PkNN) classification technique, superior to even SVM.  In this work, the challenge of cross-lingual authorship identification was tackled using a multilingual corpus consisting of 400 authors with 825 documents written in 6 different languages, which limited many popular approaches usually seen in many author identification works. First, features were not extracted by popular means of character n-grams or word frequencies as application of these feature sets on a multilingual corpus were often orthogonal to each other and would render documents written in different languages incomparable. Instead, language-independent stylometric features were extracted and passed through a feature analysis component where a set similarity search algorithm was employed to identify the top-k stylistically similar fragments (SSFs). Finally, the PkNN classifier was used on the retrieved top-k SSFs to produce the prediction as a probability mass function over all candidate authors. Comparing the results with other models, namely logistic regression (LR), naive bayes (NB), support vector machines (SVM) and convolution neural networks (CNN) as well as a previous model which also used language-independent stylometric features based on Random Forest (RF), it showed significant improvements with an impressive accuracy of 96.66%. However, multilingual author identification requires methods that are vastly different to what some models can handle which is a major cause in the disparities between accuracies between models. 

Overall, character n-grams are a widely popularized approach when representing text for author identification and are considered state-of-the-art when used as the feature set. This is due to their ability to capture nuances in lexical, syntactical, and structural level. Further, SVM sees a much higher usage and gives much greater accuracy when compared to other models. 

We will train our models using the Standardized Project Gutenberg Corpus (SPGC) [Gerlach and Font-clos 2020] (GNU General Public License) so our model can learn to classify a variety of authors over large spans of time. More specifically, we use the 2018 dataset of version SPGC-2018-07-18 which contains pre-processed data on 55905 books. 

 

## Methodology 

  

### Dataset 

The corpus used in training the SVM is a subset of the original data. More specifically, only documents that were in English and in addition, the documents that were classified as ‘text’ were selected as some were classified as ‘sound’. From this reduced corpus, only 30 authors had over 50 documents, therefore, these 30 authors were chosen with each having 50 documents, for a total of 1500 texts.  

This dataset was randomly divided into the following 3 sets: 

Training Set: 60% of dataset 

Test Set: 30% of dataset 

Validation Set: 10% of dataset 

Each set contains the 30 authors, with the training set containing 30 documents each for a total of 900 docs, the test set containing 15 documents each for a total of 450 docs and the validation set containing 5 documents each for a total of 150 docs. 

During the preprocessing stage of data, 5 page_ids did not exist in the tokenized data set of documents. Unfortunately, these missing documents are not possible to retrieve and replacing them with other authors is not possible as no other author has written close to 50 documents besides these selected. 

 

#### SVM Dataset 

+ Two SVM model were built using both datasets, dataset 1. was used for the sole purpose of training the SVM whilst dataset 2. was used to compare the BERT and SVM models. 

+ The SVM model built on dataset 1. is written in svm_original.ipynb and the SVM model built on dataset 2. is written in svm_bert_data.ipynb. 

+ The SVM model built on dataset 2. is written in svm_bert_data.ipynb uses a GPU-accelerated SVM due to the large row size of the dataset and the RAPID cuml.svm package that implements LinearSVC was used for the SVM classifier. The installation of this depends on the architecture of the operating system and the installation guide can be found [here](https://docs.rapids.ai/install?_gl=1*1wfzw4y*_ga*MTUyMjg2OTQwLjE2ODEzNTg3Mzc.*_ga_RKXFW6CM42*MTY4MTM4MjU4Ni4zLjAuMTY4MTM4MjU4Ni4wLjAuMA)

  

#### BERT Dataset 

+ As BERT model has a max 512 length limit of tokens, we split each document into small chunks with size 512. 

+ Also, the BERT model was built datasets 2. and 3., one is 2000 * 512 tokens per author, anothor is 1000 * 512 per author. 

+ The 80%, 10%, 10% dataset was split into train, test and validation dataset. 

 

### SVM Model 

Since SVM is a popular approach to tackle author identification tasks, we follow Mohsen et al. (2016) which utilizes character n-grams as features and uses a Stacked Denoising Autoencoder to further enrich these features with an SVM classifier to make final predictions. 

 

+ Character n-grams as features 

+ Used Stacked denoising autoencoders to enrich features 

+ Encoded features fed into SVM classifier 

  

These sets are first converted into a variable length character n-gram feature set of length 1-5 inclusive, where the count of each character n-gram is counted to form as initial features. 

Feature Selection is used, namely frequency-based feature selection where the top 10000 occurring n-grams are selected and the rest are removed.  

The sets are then normalized using a min-max normalization and the normalized features are then passed into a stacked denoising autoencoder (SDAE) for further enrichment of features. 

A DAE injects artificial noise in the input and attempts to output the original input by recognizing noisy data. The training of a DAE is as follows: 

Inject binomial noise into data  so that it gets transformed into , this means a random selection of elements in the dataset are set to 0 

 is encoded into the hidden representation using a non-linear transformation function . 

 

where  is the weight matrix of the encoding layer, is the bias and  is the sigmoid function, 

. 

The hidden representation is decoded into the reconstructed output . 

 

where  is the weight matrix of the decoding layer. Tied weights are used, therefore, ,  the sigmoid function is used again. 

 

The output must reconstruct the original input without noise, the reconstruction error is the cost function where binary cross-entropy was used. Further, the adam optimizer was used. 

 
 

The training of a SDAE consists of 2 procedures, unsupervised pre-training and then supervised fine-tuning. In pre-training, the steps are as follows: 

For the  DAE, the above training steps are followed to obtain the encoder function .  

This encoder function is applied on the clean input  which is fed into the next DAE as input where  

 
 

and is fed as input into the next DAE. 

The above 2 steps are repeated for each DAE, for  in the stack where  is the original input and the final  is the encoded features. 

 

Referenced from https://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf 

 

Next, the DAE’s weights are all fine-tuned. 

The fine-tuning steps are as follows: 

The encoders from each DAE are stacked and a logistic regression layer is added on top as shown in the picture below. 

The encoders have already been pre-trained following the above steps and the learnt weights are used to initialize this network. 

The fine-tuning is supervised, therefore labelled data is used to train the network and finetune the weights and parameters using categorical cross entropy as the cost function to minimize using backpropagation. Further, the logistic layer uses the softmax activation due to the classification nature of this problem and the adam optimizer is used. 

 

Referenced from https://www.hindawi.com/journals/js/2016/3632943/ 

 

After pre-training and fine-tuning the autoencoders, the final encoding function is obtained. This function can now be used to obtain the richer encoded features which can now be fed into a linear SVM classifier for predictions. 
For the SVM, a linear SVM was used with 5-fold cross validation from the Scikit-learn implementation with default parameters. 

 

The dataset texts are first converted into a variable length character n-gram feature set of length 1-5 inclusive, where the count of each character n-gram is counted to form as initial features. 

Feature Selection is used, namely frequency-based feature selection where the top 10000 occurring n-grams are selected and the rest are removed.  

The sets are then normalized using a min-max normalization and the normalized features are then passed into a stacked denoising autoencoder (SDAE) for further enrichment of features. 

A DAE injects artificial noise in the input and attempts to output the original input by recognizing noisy data. The training of a DAE is as follows: 

Inject binomial noise into data so that it gets transformed into , this means a random selection of elements in the dataset are set to 0 

  is encoded into the hidden representation using a non-linear transformation function . 

 

where is the weight matrix of the encoding layer,  is the bias and  is the sigmoid function, 

. 

The hidden representation is decoded into the reconstructed output . 

 

where  is the weight matrix of the decoding layer. Tied weights are used, therefore, ,  the sigmoid function is used again. 

The output must reconstruct the original input without noise, the reconstruction error is the cost function where binary cross-entropy was used. Further, the adam optimizer was used. 

 

The training of a SDAE consists of 2 procedures, unsupervised pre-training and then supervised fine-tuning. In pre-training, the steps are as follows: 

For the  DAE, the above training steps are followed to obtain the encoder function .  

This encoder function is applied on the clean input  which is fed into the next DAE as input where  

 
 

and  is fed as input into the next DAE. 

The above 2 steps are repeated for each DAE, for  in the stack where  is the original input and the final  is the encoded features. 

 

[2] 

Next, the DAE’s weights are all fine-tuned. 

The fine-tuning steps are as follows: 

The encoders from each DAE are stacked and a logistic regression layer is added on top as shown in the picture below. 

The encoders have already been pre-trained following the above steps and the learnt weights are used to initialize this network. 

The fine-tuning is supervised, therefore labelled data is used to train the network and finetune the weights and parameters using categorical cross entropy as the cost function to minimize using backpropagation. Further, the logistic layer uses the softmax activation due to the classification nature of this problem and the adam optimizer is used. 

 

[3] 

 

After pre-training and fine-tuning the autoencoders, the final encoding function is obtained. This function can now be used to obtain the richer encoded features which can now be fed into a linear SVM classifier for predictions. 
For the SVM, a linear SVM was used with 5-fold cross validation from the Scikit-learn implementation with default parameters. 


### BERT Model 

+ We use the Bert-base uncased model  

+ Each paragraph generates word embeddings and word position embeddings.  

+ Feed into 12 transformer layers, every layer learns connections between words by Attention Mechanism.  

+ Connect a pooling layer to aggregate information.  

+ We compare with BERT default pooling, max pooling, concatenate pooling, mean pooling. 

+ Apply a linear classifier layer in the last layer.  

+ Fine-tuning the weights is applied only on the last 4 transformer layers, the pooling layer and the classifier layer which can significantly save training time. 

  

 

SVM: 




[Table 1: Results of SVM on whole document dataset](./resources/table1.png)


[Table 2: Results of SVM on 2000 x 512 token dataset](./resources/table2.png)

[Table 3: Result](./resources/res.png)

We find that in Table 1, the SVM model achieves the highest weighted average f1-score of 96.4 on the whole document dataset when using only 1 denoising autoencoder, opposed to the paper we were following which used 3. 

On the 512 split dataset, the SVM model achieves its highest weighted average f1-score of 93.3 when using 2 denoising autoencoders, however the score increases by only a small margin compared to using an SVM with no autoencoders which achieved a weighted f1-score of 92.8. 

 

 
 

## References 

Gerlach, M. and Font-Clos, F., 2020. A standardized Project Gutenberg corpus for statistical analysis of natural language and quantitative linguistics. Entropy, 22(1), p.126.  

Kestemont, M., Manjavacas, E., Markov, I., Bevendorff, J., Wiegmann, M., Stamatatos, E., Potthast, M. and Stein, B., 2020, September. Overview of the Cross-Domain Authorship Verification Task at PAN 2020. In CLEF (Working Notes).  

Manolache, A., Brad, F., Burceanu, E., Barbalau, A., Ionescu, R. and Popescu, M., 2021. Transferring BERT-like Transformers' Knowledge for Authorship Verification. arXiv preprint arXiv:2112.05125.  

Romanov, A., Kurtukova, A., Shelupanov, A., Fedotova, A. and Goncharov, V., 2020. Authorship identification of a russian-language text using support vector machine and deep neural networks. Future Internet, 13(1), p.3.  

Sarwar, R., Li, Q., Rakthanmanon, T. and Nutanong, S., 2018. A scalable framework for cross-lingual authorship identification. Information Sciences, 465, pp.323-339. 

Mohsen, A.M., El-Makky, N.M. and Ghanem, N., 2016, December. Author identification using deep learning. In 2016 15th IEEE International Conference on Machine Learning and Applications (ICMLA) (pp. 898-903). IEEE. 

Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y., and Manzagol, P., 2010. Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion. 

Xing, C., Ma, L., and Yang, X., 2015. Stacked Denoise Autoencoder Based Feature Extraction and Classification for Hyperspectral Images. 
