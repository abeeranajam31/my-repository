#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import seaborn as sns
import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Download NLTK resources
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('Top30.csv')

# Select a subset of the dataset for analysis (data slicing)
df = df.sample(frac=0.5, random_state=42)  # For example, using 50% of the data

# Eliminate duplicate rows
df.drop_duplicates(subset=['Description'], inplace=True)

# Perform extensive text pre-processing
stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

def preprocess_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    words = re.findall(r'\w+', text.lower())  # Tokenization and lowercasing
    words = [word for word in words if word not in stop_words]  # Stop-word removal
    words = [porter.stem(word) for word in words]  # Stemming
    return ' '.join(words)

df['Processed'] = df['Description'].apply(preprocess_text)

# Feature Extraction
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['Processed'])
y = df['Query']

# Apply chi-square feature selection
chi2_selector = SelectKBest(chi2, k=300)  # Selecting top 300 features
X_kbest = chi2_selector.fit_transform(X, y)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_kbest, y, test_size=0.3, random_state=42)

# Hyperparameter Tuning for Random Forest Classifier
param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 5, 10]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Model Selection and Training
models = {
    'BernoulliNB': BernoulliNB(),
    'MultinomialNB': MultinomialNB(),
    'RandomForest': RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'], 
                                           max_depth=grid_search.best_params_['max_depth']),
    'LinearSVM': LinearSVC()
}

# Training and evaluating models
results = {}
for model_name, model in models.items():
    start_train_time = time.time()
    model.fit(X_train, y_train)
    end_train_time = time.time()
    train_time = end_train_time - start_train_time

    start_test_time = time.time()
    y_pred = model.predict(X_test)
    end_test_time = time.time()
    test_time = end_test_time - start_test_time

    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    results[model_name] = {'accuracy': accuracy, 'confusion_matrix': confusion, 'train_time': train_time, 'test_time': test_time}
    
# Plotting the results
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Accuracy Scores
axes[0].barh(list(results.keys()), [x['accuracy'] for x in results.values()])
axes[0].set_title('Accuracy Score')
axes[0].set_xlabel('Accuracy')

# Training Times
axes[1].barh(list(results.keys()), [x['train_time'] for x in results.values()])
axes[1].set_title('Training Time')
axes[1].set_xlabel('Seconds')

# Prediction Times
axes[2].barh(list(results.keys()), [x['test_time'] for x in results.values()])
axes[2].set_title('Prediction Time')
axes[2].set_xlabel('Seconds')

plt.tight_layout()
plt.show()

# Print Confusion Matrix of Bernoulli Naive Bayes Classifier output
print("\nConfusion Matrix of Bernoulli Naive Bayes Classifier:")
print(results['BernoulliNB']['confusion_matrix'])

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(results['BernoulliNB']['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.title('Confusion Matrix - Bernoulli Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Ensuring the model is correctly referenced and accessible
def real_time_prediction(text, model, tfidf_vectorizer, chi2_selector):
    try:
        # Text preprocessing
        processed = preprocess_text(text)

        # Transforming the text using the same tfidf vectorizer used during training
        vectorized = tfidf_vectorizer.transform([processed])

        # Applying Chi-squared feature selection
        selected_features = chi2_selector.transform(vectorized)

        # Predicting using the provided model
        return model.predict(selected_features)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# Example usage (replace 'svm_model' with your actual model instance)
example_text = "Handles excels sheets and data."
print("\nPredicted category:", real_time_prediction(example_text, models['LinearSVM'], tfidf, chi2_selector))
