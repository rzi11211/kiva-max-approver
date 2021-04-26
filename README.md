# Project 5 - Social Impact Group Project
## KivaMaxApprover App: Supporting Small Business Micro-Loan Field Partners in Kenya
#### Rachel Insler, Shashank Rajadhyaksha, Carlos Rivera, Precious Smith
#### *DSIR22221-E Group Project, Presented 4.26.21*


## Executive Summary

> "More than 1.7 billion people around the world are unbanked and can’t access the financial services they need. Kiva is an international nonprofit, founded in 2005 in San Francisco, with a mission to expand financial access to help underserved communities thrive." [*source*](https://www.kiva.org/about) 

They do this by crowdfunding loans and unlocking capital for the underserved, improving the quality and cost of financial services, and addressing the underlying barriers to financial access around the world. Kiva is bridges the gap between people who need and people who want to give. 

We aim to assist Kiva's field partners by building an easy-to-implement online application tool that rapidly predicts the likelihood of a prospective borrower's loan application being successful.

Using loan data collected from Kiva.org, we developed, trained, tested, and compared several classification models using the machine learning and natural language processing tools in [scikit-learn](https://scikit-learn.org/stable/) and [NLTK](https://www.nltk.org/).

Our best-performing model was able to predict loan funding status with 84.8% accuracy, using 

Our initial steps were to clean the dataset for modeling and evaluation. Our final models consisted of a numeric model and a natural language model (NLP), with each model weighted differently to predict whether an application would be funded or not. 

Columns containing numeric values were used to classify status; expired or successful. After the cleaning process numerous models were train, but our best model was Gradient Booster. A grid search found the best parameters were the following: learning_rate = 0.1, max_depth = 5, and n_estimator=100. The training accuracy (0.866) and the testing accuracy(0.848) had very little difference, indicating our numeric model had beat the null model(0.78) and was not overfit. Columns containing text were used to train our NLP model. After preprocessing and tokenizing the text, our best scoring model was a Logistic Regression with TfidfVectorizer, with combined text columns and complete tags. The best parameters for the NLP model were the following: ngram_range =(1,2), max_features=10,000, c=1.0. The training accuracy (0.837) and the testing accuracy(0.821) had very little difference, again indicating our NLP model had beat the null model(0.785) and was not overfit. The numeric model took on 70% of the weighted decision, and the NLP took 30% in determining whether an application was likely to be successful. 

On the numeric side our key findings were that features like loan_amount, fem_count, male_count, lender_term, and char_count_Tags all were positively correlated to predicting whether an application was successful. Most of these features could be seen in other numeric models as being strong indicators of loan status too. On the NLP model the strongest word combination indicators of funded loans were the following: user_favorite, single_mother, widowed, widow,  grew, and single_mom. Our theory was that investors probably thought that funding hard working women was a sound investment. 

Our online tool predicts with 85% accuracy whether or not an application is funded and will help Kiva field partners screen applications faster. Based on the numeric model the strongest predictors were loan amount, loan term, gender, number of applications, amount of text, and month request. Words made a difference with respect to the NLP model, and accurately predicted several hundred application success where the numeric model has misclassified. The predictive power of both models combined was the key to making the strongest online tool for screening efficiency. 

Moving forward, to strengthen our predictive power on the application more research regarding the combination of our two strongest models will be the focus. Enhancing our online tool to also make recommendations on how to improve loan applications is our next goal.  
 
## Problem Statement

**We aim to help Kiva Field Partners with the borrower screening process** by developing an online tool for prospective applications that will estimate the likelihood of a loan getting crowdfunded.

We will use machine learning models to **identify the key factors that predict loan funding success and develop a simple questionnaire** that allows field partners to quickly determine whether a loan application is likely to be funded or not. 


---

## Datasets

### Data Collection
This project used the data from Kiva's website for past loans that were requested (https://www.kiva.org/build/data-snapshots).

The initial list had more than a million loans and the loans ranged from 2006 onwards until 2019.  Given that older loans funding rate may have had different funding rates and a very different process, the dataset was limited to relatively recent loans starting from 2015 onwards as well as undersampled for funded loans to reduce the imbalance between funded and expired (not-funded) loans.

Using the approach, it resulted in ~419k global loans with ~350k funded and ~69k expired (non-funded) loans translating into a global funding rate of ~83% for this dataset.

The Step1 file in the 'Numeric Model' folder was used to translate the data loaded from the Kiva website to the 'kivamix.csv' file which had all the recent global applicants.

Link to file: ://www.dropbox.com/s/zuejipzviwf4s42/kivamix.csv?dl=0

Given that the loan applicants vary from country to country, the plan was to focus on one country with a relatively high volume of loans.  The top 3 countries with high volumes were Philippines (89k loans), Kenya (51k loans) and Venezuela (22k loans).  The overall global funding rate was ~83%.  Philippines had a 96%+ funding rate while Kenya had ~78% funding rate.  Given that the potential for driving higher funding rate was lower in Philippines, the plan was to focus on Kenya which had a high loan volume and a relatively lower funding rate - creating opportunities for the model to better identify who would be funded and who would not be funded.

Kenya's data reflected ~40k funded loans and ~11k expired (non-funded) loans - translating to ~78% funding rate.  Following dataset was used for building the Numeric model.
https://github.com/psmith19/kiva-max-approver/blob/main/Numeric%20Model/kivasmall.csv


### Data Dictionary
|Feature	|Type	|Dataset	|Description|
|-	|-	|-	|-	|
|ORIGINAL_LANGUAGE	|	object	|kivamix|	Native language spoken-	|
|LOAN_AMOUNT	|float64	|kivamix	|Amount of loan	|
|STATUS|int64	|kivamix	|Loan is expired(0) or loan is succesful (1)	|
|ACTIVITY_NAME	|object|kivamix	|Reason for the loan	|
|SECTOR_NAME	|object	|kivamix	|Activity group name	|
|COUNTRY_CODE	|object	|kivamix	|Country Code are all from Kenya	|
|LENDER_TERM	|float64	|kivamix	|Time until loan is payed off	|
|REPAYMENT_INTERVAL	|object	|kivamix	|Loan payment interval	|
|DISTRIBUTION_MODEL	|object	|kivamix	|Field partner or Direct	|
|word_count_DT	|int64	|kivamix	|word count in Description translation column	|
|word_count_TAGS	|int64	|kivamix	|word count in Tags column	|
|word_count_LU	|int64	|kivamix	|word count in Loan use column	|
|char_count_DT	|int64	|kivamix	|character count in Description translation column	|
|char_count_TAGS	|int64	|kivamix	|character count in Tags column	|
|char_count_LU	|int64	|kivamix	|character count in Loan use column	|
|month-	|int64	|kivamix	|month number	|
|FEM_COUNT	|float64	|kivamix	|Number of female applicants	|
|MALE_COUNT	|float64	|kivamix	|Number of male applicants	|
|PIC_TRUE_COUNT	|float64	|kivamix	|Count of borrower pictures per loan	|
|ANY_FEM	|float64	|kivamix	|Female borrowers	|
|ANY_MALE	|float64	|kivamix	|Male borrowers	|
|word_char_DT	|int64	|kivamix	|Interaction term between word and character counts for Description translation columns	|
|word_char_TAGS	|int64	|kivamix	|Interaction term between word and character counts for Tags columns	|
|word_char_LU	|int64	|kivamix	|Interaction term between word and character counts for Loan use columns	|
|MALE_FEM	|float64	|kivamix	|Interaction term between Male count and Female count	|
|MALE_PIC	|float64	|kivamix	|Male picture on application	|
|FEM_PIC	|float64	|kivamix	|Female picture on application	|


---

## Analysis

## Numeric Model Build and Analysis
### Data Cleaning Steps
- Examined the data and imputed null values with mode or 'MISSING' status in some cases.
- Converted the 3 text columns about Description, Tags, Loan Use to count of words and characters to be used for numeric modelling.  Also created an interraction variable between word and character counts given the very high correlation between word and character counts.
- Created month variable extracting from the timestamp of when the loan was posted.

### Preprocessing for Numeric model
- Both models:  Binarize target variable: {'funded' : 1, 'expired': 0})
- Numeric model:  The data was dummified for categorical variables.  In the instance where a given variable had multiple categories, the categories with fewer rows were grouped into Miscellanous group.  For example, Activity Names had some segments like 'Agriculture' had 2k+ rows which were retained, however, there were some groups like 'Wholesale' which had 16 rows and was grouped into 'Miscellaenous'.
    Also categorical data elements were dummified so that they could be used for building the model.
    
### EDA for Numeric model
- The distribution for for Loan Amount(USD) including outliers all fell into a single bin containing loan applications less than 2k amount. To take a closer look at that single bin loans-amounts a distribution for loan amounts less than 2k was created. The average loan amount funded was 326 USD less than expired(not-funded) loans. Some other findings were that the max loan was 100K.  
- The distribution plot for the lender term illustrated that the avergae funded loan had a lender term of about 12 to 13 months. Two months shorter than unsuccessful loans. 
- The average funded rate for the year was 78%, meaning no matter when you applied the probability of recieving funding was 78%. December was the highest application funded month with a probability of 89% fund rate. January had the highest count of applications, which corresponds with the Kenyan farming season. 

### Modeling, Iteration, & Evaluation of Numeric models
Given the rich features of numeric and text data in the application, we developed a Numeric model as well an NLP model.  We also then explored building a combination model bringing together the Numeric and NLP models.

In order to ensure that both models were built on identical data, we developed the Train/Test split on the NLP model and then shared the LOAN_IDs to the Numeric model - so we could score the exact same train and test loans on the 2 different models later and evaluate for accuracy.
    
#### Numeric Model Build
The numeric model was developed by scaling the numeric data given the wide range of values for data elements such as loan amount.

- Multiple models were considered and evaluated with default hyperparameters:  Logistic Regression, Decision Tree, Bagging Classifier, Random Forest Classifier, Ada Boost classifier, Gradient Boosting Classifier, SVC and Neural Network.
- The top 3 models that had a good mix of accuracy and interpretability were then narrowed down to Logistic Regression, RandomForest classifier and Gradient Boosting Classifier.  GridSearches were performed on all the 3 models to extract the optimal set of hyperparameters.
- From this exercise, Logistic Regression had an accuracy score of 0.83 for Train and Test, while Random Forest had an accuracy score of 0.85 and 0.98 (indicating overfitting.  Gradient Boost on the other hand, had an accuracy score of 0.87 and 0.85 for Train and Test - implying a better fit with not as much overfitting.  This is the recommended model from the Numeric models.
- Reviewing the classification report of the recommended Gradient Boost model, it indicated a good f1-score for funded loans but a lower f1-score for expired loans.  The low score for expired loans seems to be primarily driven by a lower recall score (more False Negatives) and represents a future opportunity from a modelling perspective.
- Some key variables that came out significant were similar across the Logistic Regression and Gradient Boost model.  Loan Amount was one of the key predictors and negatively correlated with likelihood of getting funded (i.e higher loan amount requested implied lower likelihood of getting funded).
    Similarly longer lender term was negatively correlated while having more applicants indicated higher likelihood of getting funded.  Some activities like 'Home Energy' and 'Education' were more likely to be funded while 'Retail' was less likely.
    Tags mattered - in terms of count of words as well as characters.  Month of December indiciated more chances of getting funded (potentially driven by the holiday season where more people may be in a giving mode as well as corporate initiatives).
 
## NLP Model Build & Analysis
### Data Cleaning Steps
- Parsed sampled dataset down to loans only from Kenya
- Parsed the dataset further to only the text columns needed for NLP which were:
    - TAGS, LOAN_USE & DESCRIPTION_TRANSLATED
- Removed whitespace, hyphens and hashtags from tags column so full tags will be counted as a single word when tokenized rather than separating a single tag into multiple words 
- Removed html breaks from description column
- Filled null values with empty strings to minimize null values when adding columns together
- Added text columns together in new column 'joined_text'
- Dropped remaining null values and duplicates

### Preprocessing for NLP Model
- Tokenized data to remove punctuation 
- Lemmatized data so only singular forms of words remained
- Removed English stopwords

### EDA for NLP Model
- Frequency distributions of top tags 
    - The top 3 were #parent, #womanownedbusiness, #user_favorite
- Frequency distributions of top words in all text columns combined
    - Included business, farming, farm, child
- Sentiment Analysis 
    - Did not provide much insight
    
### Modeling, Iteration & Evaluation for NLP Model
- Dataset included 51,019 observations
- X-variable: 'joined_text' column, y-variable: 'STATUS' column {'funded' : 1, 'expired': 0}
    - Baseline score: .784
- Vectorized data using Tf-IDF Vectorizer with 2 grams and 10,000 max features
    - Did not remove numbers as they gave greater predictive power
- Split data into train and test sets using train-test-split ended up with 38,264 observations for train and 12,755 for test
- Created a binary classification model using basic Logistic Regression 
    - Results:
        - Train score:  .837
        - Test score: .821
        - Cross-value score: .816
        - Accuracy score: .82
    - Word Correlations:
        - Top 5 word combinations correlated with Funded Loans: 
            - 20 000, 30 000, kes 20, singleparent, 20
        - Top 5 word combinations correlated with Expired Loans: 
            - 100 000, 100, man, repeatborrower, repairrenewreplace
    - Interpretation:
        - Our logistic regression model scored well overall; however, it didn't perform as well with the expired loans as it did with funded loans. This is likely because the classes were unbalanced. With this model we are predicting most loans to be funded and potentially giving false hope that some loans are likely to be funded when they are not. In future iterations, we would like to include a more balanced dataset, increasing the number of expired loans to be able to predict them better. 
        - The top correlated word combinations are monetary amounts in Kenyan Shillings and we can see that lower amounts under KES 30,000 are more likely to get funded and higher amounts at KES 80,000 and above are more likely to expire. Tags are also important as singleparent, repeatborrower and repairrenewreplace are all tags.
- Developed and fitted several additional classification model iterations using various hyperparameters that were optimized through trial and error. Also created model variations using stemming and modeling text columns separately to see how well they performed. Ultimately, for this project interpretation was important, so the logistic regression was the best model and was chosen for our combination model and KivaMaxApprover app.

## Combination Model
Another blended model was considered - bringing together the Numeric and NLP models.  The approach used was to take a weighted average of the probability scores from the Numeric model and the NLP model with a 70% weightage for the Numeric model and 30% weightage for the NLP model - considering that the Numeric model performed slightly better than the NLP model.

However the combined model seemed to slightly underperform the Numeric model.  It had an accuracy score that was slightly below the accuracy score of the Numeric model with opportunities again for Recall.
Having said that, there were multiple applications where the NLP model had better accuracy than the Numeric model and this represents an opportunity to further improve the Combination model - by leveraging the best parts of both the models.

## Conclusions & Future Directions  

We were able to achieve our goal of helping Kiva Field Partners with their borrower screening process by developing an online tool that quickly evaluates prospective loan applications. Our [KivaMaxApprover app](https://share.streamlit.io/psmith19/kiva-max-approver/main/kma_app.py) is built on a modified prediction algorithm that 
- returns predictions rapidly
- predicts using both Numeric & NLP models
- minimizes false positives, by requiring a "yes" from both models to predict success
- requires only five user entries (impactful features identified during the modeling process)
- requires minimal user training to operate

We believe that this app can improve efficiency for Field Partners by eliminating the need for a skilled worker to carefully review each loan application. Then, if the KivaMaxApprover app indicates that a loan is likely to be funded, an less-experienced staff member can be assigned to finalize and post the request, while more senior resources and support can be directed towards improving applications that are unlikely to be funded in their current state. 

The chart below illustrates how our "yes from both" approach performs on testing data. **9,325 of the 12,755 loans would be correctly identified by KivaMaxApprover as likely to be funded**, and so only the remaining 26.89% (3,430) of the loan applications would require an in-depth review by an experienced staffer. 

| Num. Prediction 	| NLP Predictions 	| Actual 	| Count 	|
|-	|-	|-	|-	|
| Expired 	| Expired 	| Expired 	| 645 	|
| Expired 	| Expired 	| Funded 	| 164 	|
| Expired 	| Funded 	| Expired 	| 536 	|
| Expired 	| Funded 	| Funded 	| 280 	|
| Funded 	| Expired 	| Expired 	| 228 	|
| Funded 	| Expired 	| Funded 	| 231 	|
| Funded 	| Funded 	| Expired 	| 1346 	|
| **Funded** 	| **Funded** 	| **Funded** 	| **9325** 	|


We focus on false positives to ensure that all "questionable" applications receive careful attention from the team. But the chart also illustrates some limitations of our approach: there are several hundred loans that one of our models correctly predicts as "funded," yet our app flags them for further, potentially unnecessary review. We aren't currently able to harness the full predictive power of both models, but with more time and exploration of other model combination techniques, we should be able to improve both our predictions and our app. 

Further, we'd like to enhance the functionality of our app so that it can automatically generate specific suggestions for loan application improvement, in addition to funding predictions.   

Accompanying presentation available [here](https://docs.google.com/presentation/d/1-TZQEaNkloXdCbY4gXBSgxWtonl4iISbT7ZoZXcAuUc/edit#slide=id.g5402930dac_0_462).

---

### File Structure

```
project_3_master 
|__ cleaned_data
|   |__ books_or_writing.csv   
|   |__ pre_proc_books_or_writing.csv      
|__ data
|   |__ comments_pull_complete_1616956764.csv
|   |__ comments_pull_complete_1616956764.csv
|   |__ posts_pull_complete_1616953686.csv
|   |__ posts_pull_complete_1616954678.csv
|__ provided_files
|   |__ provided_README.md
|   |__ Requirements.txt
|__ python_automation_scripts
|   |__ automation_script_comments.py
|   |__ automation_script_posts.py
|__ 1a_automation_script_posts.ipynb
|__ 1b_automation_script_comments.ipynb
|__ 2_cleaning.ipynb
|__ 3_preprocessing.ipynb
|__ 4_modeling.ipynb
|__ 5_selection_insights.ipynb
|__ presentation_project3_reddit.pdf
|__ README.md
|__ z-scratch_early_eda.ipynb