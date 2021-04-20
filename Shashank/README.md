# Project 3 - Web APIs & NLP
## Improving the auto-moderation feature for r/books and r/writing subreddits
#### Rachel Z. Insler, DSIR 22221-R

### Executive Summary

> "Every day, millions of people around the world post, vote, and comment in communities organized around their interests." [*source*](https://www.redditinc.com/) 

Reddit is a network of communities ('subredddits') based on people's interests. Users (the editors, or "redditors") are content creators as well as consumers and curators. Using a points system of upvotes and downvotes, the community determines which content and discussions are important and subsequently displayed at the top of the feed. The platform is largely self-managed, with moderators acting as forum guardians. [*source*](https://www.ionos.com/digitalguide/online-marketing/social-media/what-is-reddit/) Subreddits can use human moderators, auto-moderators, or some combination of the two. 

But with 430 million reddit users worldwide, challenges can and do arise, ranging in severity from targeted bullying and death threats to minor inconveniences such as "unfair" moderation decisions and posts that fall outside of a subreddit's stated scope. In this project, I address one of these lower-stakes issues, 'off-topic' posts, and highlight an interesting opportunity that they represent. 

Sometimes moderators (human or algorithmic) will immediately remove a post deemed 'off-topic' and other times they permit a discussion to develop around it. I propose a third option, in which a subreddit more appropriate to the attempted posts's topic is suggested to the user as an alternative. Building this recommendation engine into subreddit auto-moderation could:

1. introduce users to new subreddits of interest
2. help those subreddits to grow in size
3. help keep subreddits on-topic 
4. keep users on the reddit.com domain for longer periods of time 

Development of such a feature requires the ability to accurately determine whether or not a post intended for a subreddit actually belongs there. As a first step, I will build a model that is capable of classifying a post as belonging to one of two similar subreddits. I selected r/books and r/writing, as they have non-overlapping descriptions, but quite a bit of overlapping post content in practice. Such an autorecommendation feature could benefit both subreddits, and r/writing in particular, as its community is far smaller (1.75M members) than r/books (19.2M members).

Using submissions and comments from these two subreddits (collected with the [pushshift.io api](https://pushshift.io/api-parameters/)), I developed, trained, tested, and compared several classification models using the machine learning and natural language processing tools in [scikit-learn](https://scikit-learn.org/stable/) and [NLTK](https://www.nltk.org/).

The best-performing model was able to classify new submissions with 89.7% accuracy, using a [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) estimator on text data which had been vectorized with a [Term Frequency - Inverse Document Frequency](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) transformation. Moreover, there is potential for the model to improve with further exploration, refinement, and additional training, making it a strong candidate for inclusion in the auto-recommendation feature. 

### Problem Statement

Many user posts to reddit.com are not in keeping with the stated purpose of their intended subreddit. These posts generally lead to one of two negative outcomes: (1) 'off-topic' posts are removed by moderators, or (2) the post is permitted to remain, leading to the proliferation of more off-topic content on the thread it began.  I propose a third option, in which a subreddit more appropriate to the attempted posts's topic is suggested to the user as an alternative. Development of such a feature requires the ability to accurately determine whether or not a post intended for a subreddit actually belongs there.

**This project aims to develop a model that accurately determines whether attempted posts to reddit.com/r/writing belong on that site or are better-suited to reddit.com/r/books.**  Ultimately, this model could be incorporated into the auto-moderator algorithm on both subreddits, and used to suggest that the user post on the alternative subreddit. 


---

### Datasets

#### Data Collection

This project uses reddit.com submissions and comments collected from the **books** and **writing** subreddits using the [pushshift.io api](https://pushshift.io/api-parameters/).

Using [automation scripts](./python_automation_scripts), 120,000 unique posts were collected, 60,000 per subreddit. 

Those posts consisted of 30,000 submissions and 30,000 comments, pulled in two separate batches:

- [10,000 comments](./data/comments_pull_complete_1616956764.csv) 
- [20,000 comments](./data/comments_pull_complete_1616957780.csv)
- [10,000 submissions](./data/posts_pull_complete_1616953686.csv)
- [20,000 submissions](./data/posts_pull_complete_1616954678.csv)

#### Data Dictionary
|Feature|Type|Description|
|---|---|---|
|**subreddit**|*string*|The name of the subreddit, 'books' or 'writing'
|**text**|*string*| The post text. For comments, this is the post 'body', and for submissions, this is the post 'selftext' and 'title' merged into one string
|**created_utc**|*int*| The time (in UTC) that the post was made|
|**type**|*string*| The type of post, 'submission' or 'comment'


---

### Analysis

#### EDA & Data Cleaning Steps
1. Examined the data and removed null values

2. Deleted rows that only contained 'removed' or 'deleted' as text values

3. Created a consistent column called 'text' for both kinds of posts
- For submissions, combined selftext and title columns into one column called 'text'
- For comments, renamed the 'body' column to 'text'
   
4. Dropped duplicate posts

5. Removed all rows that had fewer than ten characters of text

6. Combined all four .csv data pulls into one large dataframe of cleaned data, saved as a [new .csv](./cleaned_data/books_or_writing.csv).

#### Preprocessing
1. Binarize target variable: {'books' : 0, 'writing': 1})

2. Removed special characters

3. Tokenized and made lowercase
   
4. Remobed English stopwords and re-checked for newly-empty rows

5. Saved as a .csv of [pre-processed data](/cleaned_data/pre_proc_books_or_writing.csv)

#### Modeling, Iteration, & Evaluation

1. Pulled a balanced sample of 5,000 observations to experiment on

2. Set up a train-test split on the cleaned and preprocessed sample training dataset. 

3. Created a basic logistic regression model with the 'text' column as the predictive feature set **X**, and 'subreddit' as the target variable **y**. 

4. Evaluated the model.
     - Results: 
       - Training Score is:  0.75675
       - Testing Score is:  0.762
       - Cross-val score is: 0.7462495308984668.

     - Interpretation: 
       - Training and testing scores were similar, indicating an acceptable level of variance, but the model is accurate only 76.2% of the time on testing data. Though outperforming the null model (50%), there is significant room for improvement.  
  
   
5.  Developed and fitted several additional classification model iterations using various hyperparameters that were optimized through trial and error. After four models, 12,500 rows were added to each class to attempt to improve predictive power. Results are summarized here:


| Model Name 	| Transformer 	| Estimator 	| Transformer  Hyperparameters 	| Estimator Hyperparameters 	| GridSearch 	| Train 	| Test 	| CV  Score 	|
|-	|:-:	|:-:	|-	|-	|-	|-	|-	|-	|
| A 	| CountVectorizer 	| LogisticRegression 	| max_features = 5_000,  max_df = .95, min_df = .05,  ngram_range=(1,2) 	| Default 	| No 	| 0.757 	| 0.762 	| 0.746 	|
| B 	| CountVectorizer 	| LogisticRegression 	| max_features = 10_000, max_df = 0.9, min_df = 2, ngram_range=(1,2) 	| C = 0.1 	| Yes 	| 0.895 	| 0.811 	| 0.784 	|
| C 	| CountVectorizer 	| LogisticRegression 	| max_features = 15_000, max_df = 0.8, min_df = 2, ngram_range=(1,2) 	| C = 0.1 	| Yes 	| 0.818 	| 0.786 	| 0.770 	|
| D 	| CountVectorizer 	| LogisticRegression 	| max_features = 10_000, max_df = 0.7, min_df = 2, ngram_range=(1,2) 	| C = 0.1 	| Yes 	| 0.816 	| 0.785 	| 0.770 	|
| E 	| CountVectorizer (added 12,500  rows to each  class) 	| LogisticRegression 	| max_features = 10_000, max_df = 0.7, min_df = 2, ngram_range=(1,2) 	| C = 0.1 	| Yes 	| 0.831 	| 0.811 	| 0.806 	|
| F 	| TFIDF 	| LogisticRegression 	| max_features = 12_500, ngram_range = (1,2) 	| C = 1 	| Yes 	| 0.889 	| 0.832 	| 0.829 	|
| G 	| TFIDF 	| LogisticRegression 	| max_features = 12_500, max_df = 0.7, min_df = 3, ngram_range = (1,3) 	| C = 1 	| Yes 	| 0.889 	| 0.831 	| 0.829 	|
| H 	| TFIDF 	| LogisticRegression 	| max_features = 10_000, max_df = 0.65, min_df = 5, ngram_range = (1,4) 	| C = 1 	| Yes 	| 0.885 	| 0.830 	| 0.829 	|
| I 	| TFIDF 	| LogisticRegression 	| max_features = 10_000, max_df = 0.65, min_df = 5, ngram_range = (1,4) 	| C = 0.1 	| No 	| 0.836 	| 0.814 	| 0.786 	|
| J 	| CountVectorizer 	| BernoulliNB 	| Default 	| Default 	| No 	| 0.828 	| 0.717 	| 0.724 	|
| K 	| CountVectorizer 	| BernoulliNB 	| max_features = 10_000, max_df = 0.7, min_df = 2, ngram_range = (1,2) 	| Default 	| No 	| 0.799 	| 0.719 	| 0.721 	|
| L 	| TFIDF 	| BernoulliNB 	| max_features = 10_000, max_df = 0.7, min_df = 2, ngram_range = (1,4) 	| Default 	| No 	| 0.792 	| 0.714 	| 0.718 	|
| M 	| CountVectorizer 	| AdaBoostClassifier with LogReg 	| max_features = 10_000, max_df = 0.7, min_df = 2, ngram_range=(1,2) 	| n_estimators=150 	| No 	| 0.813 	| 0.787 	| 0.780 	|
| N 	| CountVectorizer 	| RandomForestClassifier with Decision Tree 	| max_features = 10_000, max_df = 0.7, min_df = 2, ngram_range=(1,2) 	| n_estimators=150, max_depth = None 	| Yes 	| 0.995 	| 0.797 	| 0.795 	|
| O 	| CountVectorizer 	| ExtraTreesClassifier with DecisionTree 	| max_df = 0.7, min_df = 2, n_gram_range = (1,2) 	| n_estimators=300, max_features=auto 	| No 	| 0.997 	| 0.784 	| 0.754 	|
| P 	| CountVectorizer 	| Support Vector Machine 	| max_df = .7, min_df = 2, ngram_range=(1,2)) 	| Default 	| No 	| 0.876 	| 0.814 	| 0.807 	|

6. Identified the two best-performing models (i.e. the models with the greatest accuracy on test data without overfitting) for further evaluation

|  	| Transformer 	| Estimator 	| Transformer  Hyperparameters 	| Estimator Hyperparameters 	| Train 	| Test 	| Cross-Val Score 	|
|-	|:-:	|:-:	|-	|-	|-	|-	|-	|
| 1 	| CountVectorizer 	| LogisticRegression 	| max_features = 10_000 max_df = 0.7 min_df = 2 ngram_range=(1,2) 	| C = 0.1 	| 0.831 	| 0.811 	| 0.806 	|
| 2 	| TF-IDF 	| LogisticRegression 	| max_features = 12_500 ngram_range = (1,2) 	| C = 1 	| 0.889 	| 0.832 	| 0.829 	|

7. Trained, tested, and compared both models on a larger dataset (a balanced sample of 87,000 posts)

|  	| Transformer 	| Estimator 	| Transformer  Hyperparameters 	| Estimator Hyperparameters 	| Train 	| Test 	| Cross-Val Score 	|
|-	|:-:	|:-:	|-	|-	|-	|-	|-	|
| 1 	| CountVectorizer 	| LogisticRegression 	| max_features = 10_000 max_df = 0.7 min_df = 2 ngram_range=(1,2) 	| C = 0.1 	| 0.841 	| 0.829 	| 0.826 	|
| 2 	| TF-IDF 	| LogisticRegression 	| max_features = 12_500 ngram_range = (1,2) 	| C = 1 	| 0.845 	| 0.836 	| 0.831 	|
   - Interpretation: 
       - Adding in data increased accuracy and reduced overfitting for both models.
       - **Logistic Regression with Term Frequency-Inverse Document Frequency Vectorizer** is the winner! 

7. The selected model made correct predictions 83.6% of the time on the testing data. I observed that of the 2,864 misclassified posts, 2,189 of those were comments. 
   - Interpretation: 
       - The model is much better at classifying submissions than comments. 

8. I trained and tested the model on a submissions-only dataset.

|  	| Transformer 	| Estimator 	| Transformer  Hyperparameters 	| Estimator Hyperparameters 	| Train 	| Test 	| Cross-Val Score 	|
|-	|:-:	|:-:	|-	|-	|-	|-	|-	|
| 1 	| TF-IDF 	| LogisticRegression 	| max_features = 12_500  ngram_range = (1,2)  	| C = 1 	| 0.906 	| 0.897 	| 0.895 	|
   - Interpretation: 
       - The model's accuracy is higher for submissions than comments. At 89.7% accuracy on test data, this model correctly classifies submissions nearly nine of every ten times. 
     
 
---

### Conclusions & Recommendations 

Our model determines with 89.7% accuracy whether attempted submissions to the ‘writing’ and ‘books’ subreddits belong on those sites or are better suited for the other site. 
Ultimately, this model could be incorporated into the auto-moderator algorithm on both subreddits, and used to suggest the alternative subreddit to users.  

However, I believe the model could be improved with 
- a better understanding of its strongest predictive words 
- inight into where and why it is making classification mistakes. 

One of the key benefits of Logistic Regression is its interpretability. Words with the strongest correlation coefficients were identified for each class, and with more time, I'd be interested to investigate those further. 

To start, I recommend further analysis on the content of the misclassified posts. One line of future inquiry concerns the probabilities associated with the incorrect predictions. I identified several posts that were classified as 'writing' with high (>80% probability), but were in fact from 'books', and vice versa. What words are leading the model so confidently in the wrong direction? How can we correct these errors?

Finally, there also exists the unexplored possibility that one of the other candidate models might outperform the TF-IDF + LogReg model when it too is trained and tested on a submissions-only dataset. 

Accompanying presentation available [here](https://docs.google.com/presentation/d/14QqcgZnQoDOAsfxOTHvzoNpQ6SRuaYRRejqTdo0tPjw/edit?usp=sharing).

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
```