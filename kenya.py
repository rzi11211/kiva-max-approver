import streamlit as st
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

# loading models
num_model = pickle.load(open('./Shashank/pipe.p', 'rb'))
nlp_model = pickle.load(open('./nlp_model.p', 'rb'))


# functionS to process numeric inputs
def word_len_count(column):
    word_count = len(column.split())
    return word_count

def char_len_count(column):
    char_count = column.replace(' ','')
    char_count = len(char_count[:])
    return char_count

def feature_engineer_num(loan_amt, lend_term, description, loan_use, tags):
    if "," in loan_amt:
        loan_amnt = int("".join(loan_amt.split(",")))
    word_count_DT = word_len_count(description)
    word_count_TAGS = word_len_count(tags)
    word_count_LU = word_len_count(loan_use)
    char_count_DT = char_len_count(description)
    char_count_TAGS = char_len_count(tags)
    char_count_LU = char_len_count(loan_use)
    word_char_DT = word_count_DT*char_count_DT
    word_char_TAGS = word_count_TAGS*char_count_TAGS
    word_char_LU = word_count_LU*char_count_LU

    month = 3.87
    FEM_COUNT = 1.42
    MALE_COUNT = 2.19
    PIC_TRUE_COUNT = 2.19
    PIC_FALSE_COUNT = 0.00
    ANY_FEM = .76
    ANY_MALE = 0.98
    MALE_FEM = 9.45
    MALE_PIC = 16.105
    FEM_PIC = 9.45


    X = [loan_amnt, word_count_TAGS, lend_term, word_count_LU, char_count_DT,
        char_count_TAGS, char_count_LU, month, FEM_COUNT, MALE_COUNT,
        PIC_TRUE_COUNT, PIC_FALSE_COUNT, ANY_FEM, ANY_MALE, word_char_DT,
        word_char_TAGS, word_char_LU, MALE_FEM, MALE_PIC, FEM_PIC]
    return X

# function to process text inputs
@st.cache
def preprocess_nlp(description, loan_use, tags):
    input_list = []
    processed_tags = tags.replace(' ', '')
    processed_tags = processed_tags.replace('-', '')
    all_text = description + loan_use + processed_tags
    tokenizer = RegexpTokenizer('\w+|\$[\d.]+|S+')
    token = tokenizer.tokenize(all_text.lower())
    lemmatizer = WordNetLemmatizer()
    lem_token = [lemmatizer.lemmatize(word) for word in token]
    joined_text = ' '.join(lem_token)
    input_list.append(joined_text)
    return input_list


# user inputs
loan_amount = st.text_input("Loan Amount:")
lender_term = st.text_input("Repayment Term: # of months for repayment:")
user_description = st.text_input("Description: Tell us who you are and why you're requesting \
    a loan through Kiva.org. What will this mean for you if you're funded?:")
user_loan_use = st.text_input("Loan Use: What will your loan be used for?")
user_tags = st.text_input("Tags: Tags are helpful. Enter them here: '#myfirsttag, #mysecondtag...'")


# conditions to make sure all required input is received
if [loan_amount][0] != "":
    if [lender_term][0] != "":
        if [user_description][0] != "":
            if [user_loan_use][0] != "":
                if [user_tags][0] != "":

                    # processing inputs
                    input_num = feature_engineer_num(loan_amt=loan_amount, lend_term=lender_term,
                    description=user_description, loan_use=user_loan_use, tags=user_tags)
                    input_num = np.array(input_num).reshape(1,-1)
                    
                    predicted_status_num = num_model.predict(input_num)[0]
                    input_text = preprocess_nlp(user_description, user_loan_use, user_tags)
                    predicted_status_nlp = nlp_model.predict(input_text)[0]

                    #conditions for whether or not a loan is likely to be funded
                    if predicted_status_num >= .5:
                        if predicted_status_nlp == 1:
                            st.write(f'Your loan is likely to be funded!')
                            st.balloons()
                        else:
                            st.write(f"Your application has a low chance of being funded.\
                            For support with your application, we'd like to connect you with one of \
                            our field partners. Please contact us at xxx.xxx.xxxx")
                    else:
                        st.write(f"Your application has a low chance of being funded.\
                        For support with your application, we'd like to connect you with one of \
                        our field partners. Please contact us at xxx.xxx.xxxx")
                else:
                    st.write("Please enter at least one tag.")
            else:
                st.write("Please enter loan use.")
        else:
            st.write("Please enter a description.")
    else:
        pass
else:
    pass
