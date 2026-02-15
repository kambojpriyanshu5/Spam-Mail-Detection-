**Email Spam Detection using Machine Learning**

This project is a Spam Detection System built using Supervised Machine Learning in Python. It classifies emails as Spam or Ham (Not Spam) using the Multinomial Naive Bayes algorithm.

**Project Overview**

The model is trained on labeled email data and learns to distinguish between spam and legitimate emails. It applies Natural Language Processing (NLP) techniques to convert email text into numerical features before classification.

**Machine Learning Details**

Type of Learning: Supervised Learning
Problem Type: Binary Classification
Algorithm Used: Multinomial Naive Bayes
Text Vectorization: CountVectorizer (Bag of Words model)

**Libraries Used**

Pandas
Scikit-learn

**How Does It Work**

The dataset containing labeled emails is loaded. The data is split into training and testing sets.
Email text is converted into numerical features using CountVectorizer. a Multinomial Naive Bayes model is trained on the processed training data. the model is evaluated using Accuracy, Precision, Recall, and a Classification Report.
The trained model can classify new email text as Spam or Ham.

**Model Metrics**

Spam Detector Model Ready
Accuracy : 0.8889
Precision : 0.8214
Recall : 0.9583

**Author**

Priyanshu Kamboj
