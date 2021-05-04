from fastapi import FastAPI, APIRouter, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import similarity
# import FetchSimilarity.similarity as similarity

app = FastAPI(
    title='Fetch Rewards Cosine Similarity',
    docs_url='/',
)

router = APIRouter()

app.include_router(similarity.router, tags=['text similarity'])
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)



if __name__ == '__main__':
    uvicorn.run(app)