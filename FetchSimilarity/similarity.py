from fastapi import APIRouter, File
import numpy as np
from nltk.corpus import stopwords
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from typing import List, Dict

router = APIRouter()

np.random.seed(42)

nltk.download('stopwords')

@router.post('/similarity', response_model = Dict[str, float])
async def testSimilarity(first: str, second: str) -> dict:
    '''
    This function tests the similarity between two sentences
    Note, TFIDFVectorizor produces l2 normalized vectors, which makes
    cosine similarity equivalent to linear kernel

    first: a sentence which you would like to compare
    second: a sentence which you would like to compare

    returns a dictionary of equivalent measures including 
    the pairwise similarity, cosine similarity, and linear kernel of the input
    '''

    cleaned_sentences = cleanSentences([first, second])

    stop_words = stopwords.words('english')
    
    tfidfvectoriser = TfidfVectorizer(max_features=64, stop_words = stop_words,
                                        ngram_range=(1, 2))
    tfidfvectoriser.fit(cleaned_sentences)
    tfidf_vectors = tfidfvectoriser.transform(cleaned_sentences).toarray()
    pairwise_similarities = np.dot(tfidf_vectors,tfidf_vectors.T)
    cosine = cosine_similarity(tfidf_vectors)
    lk = linear_kernel(tfidf_vectors)

    result =  {'Pairwise Similarity': pairwise_similarities[0][1],
            'Cosine Similarity': cosine[0][1],
            'Linear Kernel': lk[0][1]}

    print(f'\n\n{result}\n\n')

    return result

def cleanSentences(sentences: List[str]) -> np.array:
    '''
    This function removes stopwords and punctuation from a list of sentences, then turns that
    list into a numpy array.

    sentences: A list of strings to be cleaned

    returns a new list of sentences
    '''

    return np.array([" ".join(re.sub(r'[^a-zA-Z]',' ', word) for word in sentence.split())
                        for sentence in sentences])
