import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import utils

def pipeline(modelA, modelB, modelC, dataset):
    device = 'cuda' if torch.device() else 'cpu'

    analyzer = utils.RestaurantTopicAnalyzer(
        similarity_threshold=0.3,
        device='mps' if torch.backends.mps.is_available() else 'cpu',
        embedding_model=modelA
    )

    results_df = analyzer.analyze_dataframe(dataset)

    # Masuk ke Sentiment Analysis
    results_df = None

    # Masuk ke Task Summarization 
    results_df = utils.pipeline_summarization(modelC, results_df)


    return None

if __name__ == '__main__':
    result = pipeline('all-MiniLM-L6-v2', None, 'google-t5/t5-small')