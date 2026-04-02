##import environment
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

#sample code

# 1. The Raw Data (Simplified)
messages = ["Free entry to win a prize", "Hey, are we still meeting?", "Claim your cash now", "Lunch at 12?"]
labels = [1, 0, 1, 0]  # 1 = Spam, 0 = Ham

# 2. Vectorization (The "Tech" Part)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# 3. Model Training
model = MultinomialNB()
model.fit(X, labels)

# 4. Inference
test_msg = ["Win cash now"]
test_vector = vectorizer.transform(test_msg)
prediction = model.predict(test_vector)

print(f"Prediction (1=Spam): {prediction[0]}")