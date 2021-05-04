from fastapi import APIRouter, File
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity

router = APIRouter()

@router.post('/similarity')
async def testSimilarity(first: str, second: str):
    '''
    This function tests the similarity between two sentences

    first: a sentence which you would like to compare
    second: a sentence which you would like to compare

    returns the cosine similarity of these functions
    '''
    nltk.download('stopwords')
    cleaned_sentences = cleanSentences([first, second])

    tfidfvectoriser = TfidfVectorizer(max_features=64)
    tfidfvectoriser.fit(cleaned_sentences)
    tfidf_vectors = tfidfvectoriser.transform(cleaned_sentences).toarray()
    
    pairwise_similarities = np.dot(tfidf_vectors,tfidf_vectors.T)

    return {'Cosine Similarity': pairwise_similarities[0][1]}

def cleanSentences(sentences: list):
    '''
    This function removes stopwords and punctuation from a list of sentences, then turns that
    list into a pandas series.

    sentences: A list of strings to be cleaned

    returns a new list of sentences
    '''

    stop_words = stopwords.words('english')

    return pd.Series([" ".join(re.sub(r'[^a-zA-Z]',' ', word).lower() for word in sentence.split() 
                            if re.sub(r'[^a-zA-Z]',' ', word).lower() not in stop_words)
                            for sentence in sentences])
