# Aidetic_assignment
I have created a model which extracts the top 10 noun chunks in  based on the TFIDF vectorizer and NER (from nltk) </br>
The Approach as follows: </br>
1. First Clean the data to remove any special characters etc, then fit a TFIDF vectorizer in it(only bigrams and trigrams).</br>
2. Then extract noun phrases using regex parser of nltk and extract different named entities like person,organization etc also using nltk.</br>
3. After extracting noun chunks we pass get importance of each noun chunks(both named entities and noun chunks from regex) based on rarity (except person named entity) of it in the trained corpus in case the word is not present in trained corpus a threshold value is assigned. For this purpose we use idf(inverse document frequency) since it gives high value to those words which occur rarely in corpus.</br>
4. A threshold is predifined (a hyperparameter) according which words are filtered i.e nouns house importance (from idf and number of times noun repeats in the document) if a noun importace is below the predined threshold then it is discarded else it is retained.</br>
5. The importance is calculated by importance= idf * ( alpha * count) count is number of times noun appears in document alpha or beta is hyper parameter which determines how much the count dominates the importance example if alpha = 0.5 then importance increse with count rate of 0.5.</br>
6. The above process for importance and filtering is applied for all NE and regex nouns except PERSON NE since in  case of the person as the person is more commonly known (more times in news) more important  he is for example if  barrack obama is persent article he will be the key word then the name of random stranger appearing same news article or the less famous journalist name in the article hence while filtering for names of person only the person who are below certain threshold are selected.</br>
7. After filtering and getting values of importances we need to select top 10 noun chunks we select by using 2 criterions  first named entities nouns are given preference over nouns extracted from regex with in named entities organization>location>GPE>person with in each categories nouns are selected based on importance.</br>
8. The meaningless words are removed by using nltk.corpus words (except names since they are not present)</br>
Finally i want summarize a the hyper pararmeters</br>
1.alpha :- how much count inflences importance values ( for all nouns except person NE)</br>
2.beta :- works same as alpha but for person NE</br>
3.thresholds:- there are various thresholds based on which nouns are filtered </br>
