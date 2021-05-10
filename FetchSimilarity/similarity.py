from fastapi import APIRouter, File
from typing import Dict
import spacy

router = APIRouter()

@router.post('/similarity', response_model = Dict[str, float])
async def testSimilarity(first: str, second: str) -> dict:
    '''
    This function tests the similarity between two sentences by using spacy to 
    convert them into word2vec vectors from the Glove model, then applying the 
    cosine similarity formula.

    Note, you will get a warning if your environment contains 'en_core_web_sm'. 
    Larger models contain word vectors, which will give better results

    first: a sentence which you would like to compare.
    second: a sentence which you would like to compare.

    returns a dictionary containing the cosine similarity of the two sentences
    '''
    
    nlp = spacy.load('en_core_web_sm')
    doc1 = nlp(first)
    doc2 = nlp(second)

    result =  {'Spacy Vector Cosine Similarity': doc1.similarity(doc2)}

    return result
    