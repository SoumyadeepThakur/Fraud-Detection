# Fraud Detection

**Goal**: Detecting fraud from credit card transcations.
**Approach**: From a statistical point of view, fraud detection methods can be broadly classified into supervised and unsupervised
methods. In supervised approach, detecting fraud is primarily a binary classification problem, where the objective is
to classify a transaction as legitimate (negative class) or fraudulent (positive class). In unsupervised fraud detection,
the problem can be thought of as an outlier detection system, assuming outlier transactions as potential instances
of fraudulent transactions. 

We explore both approaches here.

**Dataset:** We used the *UBL creditcard dataset* downloaded from [Kaggle repository](https://www.kaggle.com/mlg-ulb/creditcardfraud)

To train an autoencoder based classifier, run *code/autoencoder.py* file.

To train an MLP classifier on the encoded data obtained from an autoencoder, run *code/autoencoder.py* to train an autoencoder model, and then run *code/autoencoder.py* using the same model.
