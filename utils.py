from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from typing import List, Dict
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline
from sklearn.cluster import KMeans

class RestaurantTopicAnalyzer:
    def __init__(self, similarity_threshold=0.3, device=None, embedding_model=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = SentenceTransformer(embedding_model, device=self.device)
        self.similarity_threshold = similarity_threshold
        
        self.categories = {
            'food': (
                "discussion about food quality, taste, dishes, menu items, "
                "cooking, flavors, portions, ingredients, cuisine"
            ),
            'place': (
                "discussion about restaurant ambiance, atmosphere, decoration, "
                "location, cleanliness, seating, parking, venue"
            ),
            'price': (
                "discussion about costs, prices, value for money, expenses, "
                "affordability, budget, worth, deals"
            ),
            'service': (
                "discussion about staff behavior, waiting time, customer service, "
                "waiters, servers, attentiveness, hospitality"
            )
        }
        
        self.category_embeddings = np.array([
            self.model.encode(description, convert_to_tensor=True, device=self.device).cpu().numpy()
            for description in self.categories.values()
        ])
        
        self.kmeans = KMeans(
            n_clusters=len(self.categories),
            init=self.category_embeddings,
            n_init=1
        )
        
        self.cluster_to_category = {i: cat for i, cat in enumerate(self.categories.keys())}

    def get_topics(self, text):
        try:
            text_embedding = self.model.encode(text, convert_to_tensor=True, device=self.device).cpu().numpy()
            text_embedding = text_embedding.reshape(1, -1)
            
            cluster = self.kmeans.predict(text_embedding)[0]
            primary_category = self.cluster_to_category[cluster]
            
            similarities = {}
            for category, description in self.categories.items():
                cat_embedding = self.model.encode(description, convert_to_tensor=True, device=self.device).cpu().numpy()
                similarity = float(np.dot(text_embedding, cat_embedding) / 
                                (np.linalg.norm(text_embedding) * np.linalg.norm(cat_embedding)))
                similarities[category] = similarity
            
            relevant_topics = {
                category: score 
                for category, score in similarities.items() 
                if score > self.similarity_threshold
            }
            
            if primary_category not in relevant_topics:
                relevant_topics[primary_category] = similarities[primary_category]
            
            return relevant_topics
        
        except Exception as e:
            print(f"Error processing text: {text}")
            print(f"Error message: {str(e)}")
            return {}

    def fit(self, texts):
        embeddings = np.array([
            self.model.encode(text, convert_to_tensor=True, device=self.device).cpu().numpy()
            for text in tqdm(texts, desc="Encoding texts")
        ])
        
        self.kmeans.fit(embeddings)
        return self

    def analyze_dataframe(self, df, text_column='text'):
        self.fit(df[text_column].values)
        
        result_df = df.copy()
        
        for category in self.categories.keys():
            result_df[f'topic_{category}'] = 0
            result_df[f'score_{category}'] = 0.0
        
        result_df['topic_count'] = 0
        result_df['main_topics'] = ''
        result_df['primary_topic'] = ''
        result_df['primary_score'] = 0.0
        
        print(f"Analyzing texts on {self.device}...")
        for idx in tqdm(range(len(df)), desc="Analyzing texts"):
            text = str(df.iloc[idx][text_column])
            topics = self.get_topics(text)
            
            for category, score in topics.items():
                result_df.at[idx, f'topic_{category}'] = 1
                result_df.at[idx, f'score_{category}'] = score
            
            result_df.at[idx, 'topic_count'] = len(topics)
            result_df.at[idx, 'main_topics'] = ', '.join(topics.keys())
            
            if topics:
                primary_topic = max(topics.items(), key=lambda x: x[1])
                result_df.at[idx, 'primary_topic'] = primary_topic[0]
                result_df.at[idx, 'primary_score'] = primary_topic[1]
        
        return result_df

def extract_aspect_sentiments(dataset, aspects, classifier, max_length=50):
    for row in tqdm(dataset, desc="Processing rows", unit="row"):
        text = row['text']
        for aspect in aspects:
            # Check if the aspect needs to be analyzed
            if row[f'topic_{aspect}'] == 1:
                result = classifier(text, text_pair=aspect)[0]
                row[f'sentiment_{aspect}'] = result.get("label")
            else:
                row[f'sentiment_{aspect}'] = "Not Found"
    return dataset

def generate_chart(df, aspects):
    sentiment = ["Positive", "Negative", "Neutral", "Not Found"]
    for aspect in aspects:
        y = np.array([df[df[f'sentiment_{aspect}'] == s].shape[0] for s in sentiment])
        plt.pie(y, labels = sentiment)
        plt.legend(title = aspect)
        plt.show() 

def pipeline_sentiment_analysis(model_name, df):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f"Using device: {device}")
    device_index = 0 if device == 'cuda' else -1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device_index)

    aspects = ["food", "place", "price", "service"]

    updated_data = extract_aspect_sentiments(df.to_dict(orient='records'), aspects, classifier)
    
    generate_chart(pd.DataFrame(updated_data), aspects)
    
    df_updated = pd.DataFrame(updated_data)
    df_updated.to_csv("temp-sentiment-analysis.csv")
    return df_updated


def pipeline_summarization(df_sentiment, model='Qwen/Qwen2.5-1.5B-Instruct'):
    result = {"category":[], "sentiment":[], "summary":[]}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    qwen_model = AutoModelForCausalLM.from_pretrained(model)
    qwen_model = qwen_model.to(device)
    qwen_tokenizer = AutoTokenizer.from_pretrained(model)

    for category in df_sentiment['primary_topic'].unique():
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            prompt = "Summarize the following reviews to give out the aspects that it talks most about:\n"
            
            for text in tqdm(df_sentiment[df_sentiment['sentiment_' + category] == sentiment]['text']):
                prompt += text + "\n"
                
            full_summary = ""
            
            inputs = qwen_tokenizer(prompt, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = qwen_model.generate(**inputs, max_new_tokens=100)
            
            full_summary = qwen_tokenizer.decode(outputs[:, inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True).strip() + " "

            result['category'].append(category)
            result['sentiment'].append(sentiment)
            result['summary'].append(full_summary)
    
    return pd.DataFrame(result)