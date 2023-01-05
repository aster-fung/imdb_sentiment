# IMDB Dataset of 50K Movie Reviews
Welcome to the notebook on building a prediction model in classifying sentiments from movie reviews with TensorFlow 2.0 :D


## Dataset
This IMDB dataset was published by Maas et al for binary sentiment classification studies. It contains 50k movie reviews in English and is considered as haviing more data from some other datasets. The dataset provides 50000 highly polar movie reviews and sentiment labels (positive/negative)

For the original dataset and paper please read Learning Word Vectors for Sentiment Analysis (Maas et al., ACL2011) http://ai.stanford.edu/~amaas/data/sentiment/

## Method
Sentiment labels were transformed into binary labels (0,1). Movie reviews were tokenized with gensim functionality by mapping words to the top 5000 frequent words in the reviews. The tokenized reviews were one hot encoded and standardized. A artifical neural network with 2 hidden layers was constructed and compiled. To reduce overfitting, adam (adaptative moment estimation) was used to adjust learning rate. Dropout and early stopping were adopted as well. The resultant model has test accuracy of ~85% with >0.9 recall and >0.7 precision.

## Training curve
![train](https://github.com/aster-fung/imdb_sentiment/blob/master/loss_curve.png?raw=true)
