import streamlit as st;
import pickle


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    text = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric characters
    y = [i for i in text if i.isalnum()]
    
    # Remove stopwords and punctuation
    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    
    # Initialize PorterStemmer
    ps = PorterStemmer()
    
    # Stem the words
    text = [ps.stem(i) for i in text]
    
    # Join the processed words back into a single string
    return " ".join(text)

tfidf = pickle.load(open('vectorizer.pkl' , 'rb'))
model = pickle.load(open('model.pkl' , 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")
if st.button('Predict'):
    # 1 . preprocess

   transform_sms = transform_text(input_sms)
    # 2 . vectorizer
   vector_input = tfidf.transform([transform_sms])
    # 3 . predict
   result = model.predict(vector_input)[0]
    # 4 . Display
   if result==1:
      st.header("Spam")
   else:
      st.header("Not Spam")



