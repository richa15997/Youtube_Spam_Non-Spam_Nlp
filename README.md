# youtube_spam_non-spam_nlp

This project helps to classify youtube comments as spam or non spam including emoticons. This is done using natural language processing techniques in python.

# Requirements

Large dataset was collected from UCI dataset. 

# Libraries

Pandas library is used. Natural Language Tookit(nltk) is used to pre-process the data. 

# Details

All the dataset was combined into a single tsv file.

As the dataset also contains many emoticons, advanced pre-processing is done.
2 approaches were taken
1) replace the emoticons with text from a list by using regular expressions.
2) while using stopwords leave all characters which form words or combine to form emoticons, then use the Twitter Tokenizer to vectorize.

