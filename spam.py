from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import pandas as pd

# Load Dataset
def load_data():
    data = pd.read_csv("C:\\Users\\PRIYANSHU KAMBOJ\\Desktop\\Spam-detection-main\\email_classification.csv")
    return data

# Train Model
def train_model(data):
    X = data['email']
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="spam")
    recall = recall_score(y_test, y_pred, pos_label="spam")

    report = classification_report(y_test, y_pred)

    return model, vectorizer, accuracy, precision, recall, report

# Prediction Function
def predict_email(model, vectorizer, email_text):
    input_features = vectorizer.transform([email_text])
    prediction = model.predict(input_features)
    return prediction[0]

# Main Program
if __name__ == "__main__":
    data = load_data()
    model, vectorizer, accuracy, precision, recall, report = train_model(data)

    print("\nSpam Detector Model Ready!")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}\n")

    print("Classification Report:\n")
    print(report)

    while True:
        email_input = input("\nEnter email text (or type 'exit' to quit): ")

        if email_input.lower() == "exit":
            print("Exiting Spam Detector...")
            break

        prediction = predict_email(model, vectorizer, email_input)

        if prediction == "ham":
            print(" This email is classified as: Ham (Not Spam)")
        else:
            print(" This email is classified as: Spam")
