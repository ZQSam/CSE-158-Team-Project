import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline

# Load the dataset
splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/GonzaloA/fake_news/" + splits["train"])

# Preprocess the text
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()  # Lowercase all words
        text = ''.join(char for char in text if char.isalnum() or char.isspace())  # Remove punctuation
    return text

df['text'] = df['text'].apply(preprocess_text)

# Drop rows with missing or invalid text entries
df = df[df['text'].notna() & (df['text'] != '')]

# Drop rows with missing values in the 'text' or 'label' columns
df.dropna(subset=['text', 'label'], inplace=True)

# Split the data into train, validation, and test sets
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)

# Define features and labels
X_train, y_train = train_df['text'], train_df['label']
X_val, y_val = val_df['text'], val_df['label']
X_test, y_test = test_df['text'], test_df['label']

# Method 1: Bag-of-Words with Logistic Regression
bow_vectorizer = CountVectorizer(max_features=1000)
bow_logistic_model = Pipeline([
    ('vectorizer', bow_vectorizer),
    ('classifier', LogisticRegression(C=1.0, class_weight='balanced', random_state=42))
])

# Train the BoW model
bow_logistic_model.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = bow_logistic_model.predict(X_val)

bow_results = [
    "Bag-of-Words Logistic Regression:",
    f"Accuracy: {accuracy_score(y_val, y_val_pred)}",
    f"Precision: {precision_score(y_val, y_val_pred)}",
    f"Recall: {recall_score(y_val, y_val_pred)}",
    f"F1-Score: {f1_score(y_val, y_val_pred)}",
    f"Balanced Accuracy: {balanced_accuracy_score(y_val, y_val_pred)}\n"
]

# Method 2: TF-IDF with Logistic Regression
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_logistic_model = Pipeline([
    ('vectorizer', tfidf_vectorizer),
    ('classifier', LogisticRegression(C=1.0, class_weight='balanced', random_state=42))
])

# Train the TF-IDF model
tfidf_logistic_model.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred_tfidf = tfidf_logistic_model.predict(X_val)

tfidf_results = [
    "TF-IDF Logistic Regression:",
    f"Accuracy: {accuracy_score(y_val, y_val_pred_tfidf)}",
    f"Precision: {precision_score(y_val, y_val_pred_tfidf)}",
    f"Recall: {recall_score(y_val, y_val_pred_tfidf)}",
    f"F1-Score: {f1_score(y_val, y_val_pred_tfidf)}",
    f"Balanced Accuracy: {balanced_accuracy_score(y_val, y_val_pred_tfidf)}\n"
]

# Method 3: Hyperparameter Tuning for Logistic Regression
best_C = None
best_balanced_accuracy = 0

for C in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
    model = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=1000)),
        ('classifier', LogisticRegression(C=C, class_weight='balanced', random_state=42))
    ])
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    balanced_acc = balanced_accuracy_score(y_val, y_val_pred)
    if balanced_acc > best_balanced_accuracy:
        best_C = C
        best_balanced_accuracy = balanced_acc

best_C_results = [
    f"Best C Value: {best_C}",
    f"Best Balanced Accuracy on Validation Set: {best_balanced_accuracy}\n"
]

# Train the final model with the best C value
final_model = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=1000)),
    ('classifier', LogisticRegression(C=best_C, class_weight='balanced', random_state=42))
])
final_model.fit(X_train, y_train)

# Evaluate on the test set
y_test_pred = final_model.predict(X_test)

final_model_results = [
    "Hyperparameter Tuning for Logistic Regression:",
    f"Accuracy: {accuracy_score(y_test, y_test_pred)}",
    f"Precision: {precision_score(y_test, y_test_pred)}",
    f"Recall: {recall_score(y_test, y_test_pred)}",
    f"F1-Score: {f1_score(y_test, y_test_pred)}",
    f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_test_pred)}"
]

# Write all results to a txt file
with open('model_results.txt', 'w') as f:
    for result in bow_results + tfidf_results + best_C_results + final_model_results:
        f.write(result + '\n')
