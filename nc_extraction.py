# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:09:52 2021

@author: kaush
"""
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import nltk
import re 
from nltk.corpus import words
import operator
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

nltk.download('stopwords')
nltk.download('words')
nltk.download('maxent_ne_chunker')
nltk.download('brown')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def clean_text(text):
  return re.sub(r'[^A-Za-z0-9 ]+', ' ', text)


class Noun_extractor:
  ''' The object extracts nouns phrases from a article given acoording to the importance calculated using TFIDF values of the words and NER
     below is the initialization part of the where we have to give tfidf vectorizer trained on the corpous (only bi and trigram) and various thresholds
     threshold regex: is threshold score for nouns  which are not recognized as any named enitity(like Person ,Organization etc) to be considered important
     threshold person and other ne is the thresholds for named entities like organization etc to be considered important
     alpha and beta is used to compute the increment in importance of words as its count increses in article or set of article'''
  def __init__(self,vectorizer,threshold_person=4,threshold_regex=4.5,threshold_other_ne=4.5,alpha=0.9,beta=0.5):
    self.vectorizer=vectorizer
    self.threshold_person=threshold_person
    self.threshold_regex=threshold_regex
    self.threshold_other_ne=threshold_other_ne
    self.alpha =alpha
    self.beta=beta
    self.text=''
  
  def extract_np(self,psent):
    ''' extracts nouns using regex parser of nltk '''
    grammar = r"""
      NP: {<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}                # chunk sequences of proper nouns
      {<NN>+}                 # chunk consecutive nouns
      """
    cp = nltk.RegexpParser(grammar)
    psent = cp.parse(psent)
    for subtree in psent.subtrees():
      if subtree.label() == 'NP':
       yield ' '.join(word for word, tag in subtree.leaves())

  def extract_org(self,psent):
    ''' extracts org named entity from a given tree'''
    for subtree in psent.subtrees():
     if subtree.label() == 'ORGANIZATION':
      yield ' '.join(word for word, tag in subtree.leaves())
  
  def extract_person(self,psent):
    ''' extracts person named entity from a given tree'''
    for subtree in psent.subtrees():
     if subtree.label() == 'PERSON':
      yield ' '.join(word for word, tag in subtree.leaves())

  def extract_facility(self,psent):
    ''' extracts facility named entity from a given tree'''
    for subtree in psent.subtrees():
     if subtree.label() == 'FACILITY':
      yield ' '.join(word for word, tag in subtree.leaves())


  def extract_gpe(self,psent):
    ''' extracts GPE named entity from a given tree'''
    for subtree in psent.subtrees():
     if subtree.label() == 'GPE':
      yield ' '.join(word for word, tag in subtree.leaves())


  def extract_location(self,psent):
    ''' extracts location named entity from a given tree'''
    for subtree in psent.subtrees():
     if subtree.label() == 'LOCATION':
      yield ' '.join(word for word, tag in subtree.leaves())

  def phrase_for_ne(self):
    # Tag sentence for part of speech
    tagged_sent = pos_tag(self.text.split())  # List of tuples with [(Word, PartOfSpeech)]
    parsed_sent = nltk.ne_chunk(tagged_sent)
    return parsed_sent

  def pharse(self):

    tagged_sent = pos_tag(self.text.split())
    return tagged_sent

  def list_the_nouns(self,items):
    ''' creates the list of nouns from given generator'''
    object_list=[]
    for item in items:
     if len(item.split())>1 and len(item.split())<4:
        object_list.append(item.lower())
    return object_list
  
  def filter_normal_ne(self,objects):
    ''' assigning importance to all named entities except person type'''
    final_list={}
    for obj in objects:
      if obj not in list(self.vectorizer.vocabulary_.keys()):
        final_list[obj.lower()]=self.threshold_other_ne
      else:
        index=self.vectorizer.vocabulary_[obj]
        idf_val=self.vectorizer.idf_[index]
        count=len(re.findall(str(obj), self.text, re.I))
        if count>1:
          count=count*self.alpha
          value=count*idf_val
        else:
          value=idf_val
        if value>=self.threshold_other_ne:
          final_list[obj.lower()]=value
    return final_list

  def filter_normal_noun(self,objects):
    ''' Assigning importance to nouns which are not considered named entities'''
    final_list={}
    for obj in objects:
      if obj not in list(self.vectorizer.vocabulary_.keys()):
        final_list[obj.lower()]=self.threshold_other_ne
      else:
        index=self.vectorizer.vocabulary_[obj]
        idf_val=self.vectorizer.idf_[index]
        count=len(re.findall(str(obj), self.text, re.I))
        if count>1:
          count=count*self.alpha
          value=count*idf_val
        else:
          value=idf_val
        if value>=self.threshold_regex:
          final_list[obj.lower()]=value
    return final_list 

  def filter_persons(self,objects):
    '''Assigning importance to all persons in text'''
    final_list={}
    for obj in objects:
      if obj not in list(self.vectorizer.vocabulary_.keys()):
        continue
      index=self.vectorizer.vocabulary_[obj]
      idf_val=self.vectorizer.idf_[index]
      value=idf_val
      count=len(re.findall(str(obj), self.text, re.I))
      if count>1:
        value=value/(self.beta*count)
      if value<=self.threshold_person:
        final_list[obj.lower()]=value
    return final_list

  def get_noun_chunks(self,text):
    '''code to get noun chunks by using named entity and normal regex '''
    self.text=text
    tags=self.pharse()
    nouns_from_regex=self.list_the_nouns(self.extract_np(tags))
    tags_of_ne=self.phrase_for_ne()
    list_of_organizations=self.list_the_nouns(self.extract_org(tags_of_ne))
    list_of_persons=self.list_the_nouns(self.extract_person(tags_of_ne))
    list_of_gpe=self.list_the_nouns(self.extract_gpe(tags_of_ne))
    list_of_locations=self.list_the_nouns(self.extract_location(tags_of_ne))
    return {'organizations':list_of_organizations,'persons':list_of_persons,'gpe':list_of_gpe,'locations':list_of_locations,'others':nouns_from_regex}

  def get_importance_of_nouns(self,noun_dict):
    '''Assigning importances to various types of named entities '''
    org_dict=self.filter_normal_ne(noun_dict['organizations'])
    person_dict=self.filter_persons(noun_dict['persons'])
    gpe_dict=self.filter_normal_ne(noun_dict['gpe'])
    locations_dict=self.filter_normal_ne(noun_dict['locations'])
    others_dict=self.filter_normal_noun(noun_dict['others'])
    return {'organization':org_dict,'person_dict':person_dict,'gpe_dict':gpe_dict,'locations_dict':locations_dict,'others_dict':others_dict}
  
  def check_meaningful(self,text):
    ''' Code to check if words are meaningful (do not check names) '''
    from nltk.corpus import words
    for i in text.split():
      if len(i)<2:
        return False
    for i in text.split():
      if i in words.words():
        return True
    return False

  def filter_produce_top_10(self,dict_of_imp):
    '''Return top 10 nouns
      format first return organization named entities (most imp) next other non person named entities then person then regex extracted nouns
      order of importance
      org>other_ne>person_ne>regex_nouns
        '''
    list_of_nouns=[]

    for noun,importance in dict(sorted(dict_of_imp['locations_dict'].items(), key=operator.itemgetter(1),reverse=True)).items():
      if self.check_meaningful(noun) and (noun not in list_of_nouns):
          if len(list_of_nouns)<10:
            list_of_nouns.append(noun)
          else:
            return list_of_nouns
    if len(list_of_nouns)>=10:
      return list_of_nouns
    

    for noun,importance in dict(sorted(dict_of_imp['organization'].items(), key=operator.itemgetter(1),reverse=True)).items():
      if self.check_meaningful(noun) and (noun not in list_of_nouns):
          if len(list_of_nouns)<10:
            list_of_nouns.append(noun)
          else:
            return list_of_nouns
    if len(list_of_nouns)>=10:
      return list_of_nouns

    for noun,importance in dict(sorted(dict_of_imp['gpe_dict'].items(), key=operator.itemgetter(1),reverse=True)).items():
      if self.check_meaningful(noun) and (noun not in list_of_nouns):
          if len(list_of_nouns)<10:
            list_of_nouns.append(noun)
          else:
            return list_of_nouns
    if len(list_of_nouns)>=10:
      return list_of_nouns


    for noun,importance in dict(sorted(dict_of_imp['person_dict'].items(), key=operator.itemgetter(1))).items():
      if (noun not in list_of_nouns):
          if len(list_of_nouns)<10:
            list_of_nouns.append(noun)
          else:
            return list_of_nouns
    if len(list_of_nouns)>=10:
      return list_of_nouns
    
    for noun,importance in dict(sorted(dict_of_imp['others_dict'].items(), key=operator.itemgetter(1))).items():
      if self.check_meaningful(noun) and (noun not in list_of_nouns):
          if len(list_of_nouns)<10:
            list_of_nouns.append(noun)
          else:
            return list_of_nouns

    return list_of_nouns

text_data=pd.read_excel('data.xlsx')['Text'].apply(clean_text)
vector=TfidfVectorizer(ngram_range=(2,3))
vector.fit(text_data)
noun_obj=Noun_extractor(vector)
chunks=noun_obj.get_noun_chunks(text_data[60])
importances=noun_obj.get_importance_of_nouns(chunks)
print(noun_obj.filter_produce_top_10(importances))
with open('vector.pkl', 'wb') as output:  # Overwrites any existing file.
    pickle.dump(vector, output)
