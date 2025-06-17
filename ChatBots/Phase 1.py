from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string

nltk.download('punkt')

corpus = [
    "Hi there!! How can I help you?",
    "We provide support for internet and mobile issues.",
    "You can reach us from 9 AM to 6 PM every day.",
    "Goodbye! Have a great day.",
    "Please restart your modem and try again.",
    "Check if your router cables are plugged in properly.",
    "Sure! I can help you with your billing issues.",
    "Let me connect you with a human support agent."
]

vec = TfidfVectorizer()
X = vec.fit_transform(corpus)
def get_response(query):
    query_vec = vec.transform([query])
    similarity = cosine_similarity(query_vec, X)
    print(similarity)
    best_idx = similarity.argmax()
    best_score = similarity[0, best_idx]
    
    if best_score < 0.2:
        return "I'm sorry, I didn't understand that. Can you rephrase?"
    return corpus[best_idx]

print("Bot: Hello! Type 'exit' to stop chatting.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break
    response = get_response(user_input)
    print("Bot:", response)






