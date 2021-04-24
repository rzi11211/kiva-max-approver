# import libraries
import streamlit as st
import pickle
import numpy as np
from PIL import Image
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

#header
st.title('KivaMaxApprover')

# header image
col1, col2 = st.beta_columns(2)
image1 = Image.open('./images/field.jpeg')
col1.image(image1, use_column_width=True)
image2 = Image.open('./images/ginger.jpeg')
col2.image(image2, use_column_width=True)

# user inputs
loan_amount = st.text_input("Enter Loan Amount:* (ex. 10,000)")
lender_term = st.text_input("Enter Number of Months for Repayment:*")
user_description = st.text_input("Enter Description:*  (Tell us who the borrower is and \
    why they're requesting a loan through Kiva.org)")
user_loan_use = st.text_input("Enter Loan Use:* (ex. To purchase...)")
user_tags = st.text_input("Enter Tags:*  (ex. #myfirsttag, #mysecondtag...)")


# functions to process user inputs for numeric model
# creating features with word count of text inputs
def word_len_count(column):
    word_count = len(column.split())
    return word_count
# creating features with character count of text inputs
def char_len_count(column):
    char_count = column.replace(' ','')
    char_count = len(char_count[:])
    return char_count
# creating list of features for X variable, returns X variable for numeric model
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
    # default feature values
    month = 3
    FEM_COUNT = 1
    MALE_COUNT = 1
    PIC_TRUE_COUNT = 1
    PIC_FALSE_COUNT = 0
    ANY_FEM = 1
    ANY_MALE = 1
    MALE_FEM = 1
    MALE_PIC = 1
    FEM_PIC = 1
    # defining X variable for numeric model
    X = [loan_amnt, lend_term, word_count_DT, word_count_TAGS, word_count_LU,
        char_count_DT, char_count_TAGS, char_count_LU, word_char_DT, word_char_TAGS,
        word_char_LU, month, FEM_COUNT, MALE_COUNT, PIC_TRUE_COUNT, PIC_FALSE_COUNT,
        ANY_FEM, ANY_MALE, MALE_FEM, MALE_PIC, FEM_PIC]
    return X

# function to preprocess user inputs for nlp model
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


# loading models
num_model = pickle.load(open('./models/numeric_model.p', 'rb'))
nlp_model = pickle.load(open('./models/nlp_model.p', 'rb'))


# condition to make sure all required input is received
if [loan_amount][0] == "" or [lender_term][0] == "" or [user_description][0] == "" or [user_loan_use][0] == "" or [user_tags][0] == "":
    st.write("*Please answer all questions.")
else:
    # processing inputs for numeric model
    input_num = feature_engineer_num(loan_amt=loan_amount, lend_term=lender_term,
    description=user_description, loan_use=user_loan_use, tags=user_tags)
    input_num = np.array(input_num).reshape(1,-1)
    predicted_status_num = num_model.predict(input_num)[0]

    # processing inputs for nlp model
    input_text = preprocess_nlp(user_description, user_loan_use, user_tags)
    predicted_status_nlp = nlp_model.predict(input_text)[0]

    #conditions for whether or not a loan is likely to be funded
    # must be predicted as funded by both models to receive positive result
    if predicted_status_num >= .5 and predicted_status_nlp == 1:
        st.write(f'This loan is likely to be funded! Proceed to next steps.')
        st.balloons()
    else:
        st.write(f"This application has a low chance of being funded.\
        Please consider making adjustments.")
