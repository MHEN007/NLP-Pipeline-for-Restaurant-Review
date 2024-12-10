from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from typing import List, Dict
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class RestaurantTopicAnalyzer:
    def __init__(self, similarity_threshold = 0.3, device = None, embedding_model=None):
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
        
        self.category_embeddings = {
            category: self.model.encode(description, convert_to_tensor=True, device=self.device)
            for category, description in self.categories.items()
        }

    def get_topics(self, text):
        try:
            text_embedding = self.model.encode(text, convert_to_tensor=True, device=self.device)
            
            similarities = {}
            for category, cat_embedding in self.category_embeddings.items():
                text_embedding = text_embedding.to(self.device)
                cat_embedding = cat_embedding.to(self.device)
                
                similarity = float(torch.cosine_similarity(text_embedding, cat_embedding, dim=0))
                similarities[category] = similarity
            
            relevant_topics = {
                category: score 
                for category, score in similarities.items() 
                if score > self.similarity_threshold
            }
            
            if not relevant_topics:
                max_category = max(similarities.items(), key=lambda x: x[1])
                relevant_topics = {max_category[0]: max_category[1]}
            
            return relevant_topics
        
        except Exception as e:
            print(f"Error processing text: {text}")
            print(f"Error message: {str(e)}")
            return {}

    def analyze_dataframe(self, df, text_column = 'text'):
        result_df = df.copy()
        
        for category in self.categories.keys():
            result_df[f'topic_{category}'] = 0
            result_df[f'score_{category}'] = 0.0
        
        result_df['topic_count'] = 0
        result_df['main_topics'] = ''
        result_df['primary_topic'] = ''
        result_df['primary_score'] = 0.0
        
        print(f"Analyzing texts on {self.device}...")
        for idx in tqdm(range(len(df))):
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

    def generate_analysis_report(self, df):
        topic_columns = [col for col in df.columns if col.startswith('topic_') and col != 'topic_count']
        
        topic_distribution = {
            col.replace('topic_', ''): df[col].sum()
            for col in topic_columns
        }
        
        primary_topic_dist = df['primary_topic'].value_counts().to_dict()
        
        avg_scores = {
            col.replace('score_', ''): df[col].mean()
            for col in df.columns if col.startswith('score_')
        }
        
        return {
            'total_texts': len(df),
            'topic_distribution': topic_distribution,
            'primary_topic_distribution': primary_topic_dist,
            'avg_scores': avg_scores,
            'avg_topics_per_text': df['topic_count'].mean()
        }

    def plot_topic_distribution(self, df):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        topic_columns = [col for col in df.columns if col.startswith('topic_') and col != 'topic_count']
        topic_counts = [df[col].sum() for col in topic_columns]
        topic_names = [col.replace('topic_', '') for col in topic_columns]
        
        sns.barplot(x=topic_names, y=topic_counts)
        plt.title('All Topics Distribution')
        plt.xlabel('Topics')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        primary_topic_counts = df['primary_topic'].value_counts()
        sns.barplot(x=primary_topic_counts.index, y=primary_topic_counts.values)
        plt.title('Primary Topic Distribution')
        plt.xlabel('Topics')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

def pipeline_summarization(model, df_sentiment):
    result = {"category":[], "sentiment":[], "summary":[]}

    t5_small_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    t5_small_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

    for category in df_sentiment['category'].unique():
    # Get each sentiment for the category
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            prompt = "Summarize: "
            
            for text in tqdm(df_sentiment[df_sentiment['sentiment_' + category] == sentiment]['text']):
                prompt += text + "\n"
                
            # text_chunks = [prompt[i:i + 512] for i in range(0, len(prompt), 512)]
            full_summary = ""
            
            # for chunk in text_chunks:
            inputs = t5_small_tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True).to('gpu' if torch.device() else 'cpu')
            
            with torch.no_grad():
                outputs = t5_small_model.generate(**inputs, max_length=50, min_length=10, 
                                                        length_penalty=2.0, num_beams=4, 
                                                        early_stopping=True)
            
            full_summary += t5_small_tokenizer.decode(outputs[0], skip_special_tokens=True).strip() + " "

            result['category'].append(category)
            result['sentiment'].append(sentiment)
            result['summary'].append(full_summary)