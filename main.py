import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import utils

def pipeline(modelA, modelB, modelC, dataset):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    analyzer = utils.RestaurantTopicAnalyzer(
        similarity_threshold=0.3,
        device=device,
        embedding_model=modelA
    )
    
    dataset = pd.read_csv(dataset)

    results_df = analyzer.analyze_dataframe(dataset)

    # Masuk ke Sentiment Analysis
    results_df = utils.pipeline_sentiment_analysis(modelB, results_df)

    # Masuk ke Task Summarization 
    results_df = utils.pipeline_summarization(results_df, modelC)


    return None

if __name__ == '__main__':
    result = pipeline('all-MiniLM-L6-v2', 'yangheng/deberta-v3-base-absa-v1.1', 'Qwen/Qwen2.5-1.5B-Instruct', './Datasets/dataset.csv')