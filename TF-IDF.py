# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 14:52:59 2018

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 02:29:05 2018

@author: Dell
"""
import nltk
import re
import heapq
import numpy as np


sherlock="""Dr. Watson, Mr. Sherlock Holmes," said Stamford, introducing us.

"How are you?" he said cordially, gripping my hand with a strength for which I should hardly have given him credit. "You have been in Afghanistan, I perceive."

"How on earth did you know that?" I asked in astonishment.

"Never mind," said he, chuckling to himself. "The question now is about haemoglobin. No doubt you see the significance of this discovery of mine?"

"It is interesting, chemically, no doubt," I answered, "but practically----"

"Why, man, it is the most practical medico-legal discovery for years. Don't you see that it gives us an infallible test for blood stains? Come over here now!" He seized me by the coat-sleeve in his eagerness, and drew me over to the table at which he had been working. "Let us have some fresh blood," he said, digging a long bodkin into his finger, and drawing off the resulting drop of blood in a chemical pipette. "Now, I add this small quantity of blood to a litre of water. You perceive that the resulting mixture has the appearance of pure water. The proportion of blood cannot be more than one in a million. I have no doubt, however, that we shall be able to obtain the characteristic reaction." As he spoke, he threw into the vessel a few white crystals, and then added some drops of a transparent fluid. In an instant the contents assumed a dull mahogany colour, and a brownish dust was precipitated to the bottom of the glass jar.

"Ha! ha!" he cried, clapping his hands, and looking as delighted as a child with a new toy. "What do you think of that?"

"It seems to be a very delicate test," I remarked.

"Beautiful! beautiful! The old guaiacum test was very clumsy and uncertain. So is the microscopic examination for blood corpuscles. The latter is valueless if the stains are a few hours old. Now, this appears to act as well whether the blood is old or new. Had this test been invented, there are hundreds of men now walking the earth who would long ago have paid the penalty of their crimes."""

dataset=nltk.sent_tokenize(sherlock)

for i in range(len(dataset)):
    dataset[i]=dataset[i].lower()
    dataset[i]=re.sub(r'\W',' ',dataset[i])
    dataset[i]=re.sub(r'\s+',' ',dataset[i])
#Creating histogram
wordcount={}
for data in dataset:
    words=nltk.word_tokenize(data)
    for word in words:
        if word not in wordcount.keys():
            wordcount[word]=1
        else:
            wordcount[word]+=1
    
freq_words=heapq.nlargest(100,wordcount,key=wordcount.get)  

#IDF-Matrix
words_idf={}
for word in freq_words:
    doc_count=0
    for data in dataset:
        if word in nltk.word_tokenize(data):
            doc_count+=1
    words_idf[word]=np.log((len(dataset)/doc_count)+1)

#TF-Matrix
tf_matrix={}
for word in freq_words:
    doc_tf=[]
    for data in dataset:
        frequency=0
        for w in nltk.word_tokenize(data):
            if w==word:
                frequency+=1
        tf_word=frequency/len(nltk.word_tokenize(data))
        doc_tf.append(tf_word)
    tf_matrix[word]=doc_tf
#TF-IDF Calculation
tfidf_matrix=[]
for word in tf_matrix.keys():
    tfidf=[]
    for value in tf_matrix[word]:
        score=value*words_idf[word]
        tfidf.append(score)
    tfidf_matrix.append(tfidf)

X=np.asarray(tfidf_matrix)
X=np.transpose(X)