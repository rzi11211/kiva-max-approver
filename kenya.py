import streamlit as st
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer



# @st.cache
def preprocess_nlp(description, tags):
    input_list = []
    processed_tags = tags.replace(' ', '')
    processed_tags = processed_tags.replace('-', '')
    all_text = description + processed_tags
    tokenizer = RegexpTokenizer('\w+|\$[\d.]+|S+')
    token = tokenizer.tokenize(all_text.lower())
    lemmatizer = WordNetLemmatizer()
    lem_token = [lemmatizer.lemmatize(word) for word in token]
    joined_text = ' '.join(lem_token)
    input_list.append(joined_text)
    return input_list

nlp_model = pickle.load(open('./nlp_model.p', 'rb'))

user_description = st.text_input("Tell us who you are and why you're requesting a loan through Kiva.org:")
user_tags = st.text_input("Tags are helpful. Enter them here: '#myfirsttag, #mysecondtag...'")
# st.write([user_description][0])
# st.write([user_tags][0])

if [user_description][0] != "":
    if [user_tags][0] != "":
        input_text = preprocess_nlp(user_description, user_tags)

        # nlp_model = pickle.load(open('./nlp_model.p', 'rb'))

        predicted_status = nlp_model.predict(input_text)[0]
        if predicted_status == 1:
            st.write(f'Your loan is likely to be funded!')
            st.balloons()
        else:
            st.write(f"Your application is has a low chance of being funded.\
            For support with your application, we'd like to connect you with one of \
            our field partners. Please contact us at xxx.xxx.xxxx")
    else:
        st.write("Please enter at least one tag.")
else:
    pass
