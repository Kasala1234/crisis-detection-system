import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from preprocess import clean_text

df = pd.read_csv("data/train.csv")

df["text"] = df["text"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["target"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)

print(classification_report(y_test, preds))

joblib.dump(model, "models/baseline_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")