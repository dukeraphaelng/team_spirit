For Team Spirits project we tackle author identification using 2 models, one based on SVM and one based on BERT

We used two datasets:
1. The top 30 authors with 50 documents each with a 60, 30, 10% split on the train, test and validation set respectively, https://unsw-my.sharepoint.com/:f:/g/personal/z5316653_ad_unsw_edu_au/Eol-3sqFPrhLsBt_ifuuF7kBv_XXpSkD8F1QB3HoI3clsA?e=J72fXn
2. The same 30 authors with 200*512 tokens each with a 80, 10, 10% split on the train, test and validation set respectively, https://unsw-my.sharepoint.com/:u:/g/personal/z5408671_ad_unsw_edu_au/ER0ph37G5dNOr7mX3WnOFKMBXhu_FFc7e-6eijhxJPcuhA?e=9fmPOs

Two SVM model were built using both datasets, dataset 1. was used for the sole purpose of training the SVM whilst dataset 2. was used to compare the BERT and SVM models.
The SVM model built on dataset 1. is written in svm_original.ipynb and the SVM model built on dataset 2. is written in svm_bert_data.ipynb.

The SVM model built on dataset 2. uses a GPU-accelerated SVM due to the large row size of the dataset and the RAPID cuml.svm package that implements LinearSVC was used for the SVM classifier. The installation of this depends on the architecture of the operating system and the installation guide can be found here https://docs.rapids.ai/install?_gl=1*1wfzw4y*_ga*MTUyMjg2OTQwLjE2ODEzNTg3Mzc.*_ga_RKXFW6CM42*MTY4MTM4MjU4Ni4zLjAuMTY4MTM4MjU4Ni4wLjAuMA..
