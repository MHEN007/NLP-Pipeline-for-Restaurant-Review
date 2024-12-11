from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from typing import List, Dict
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline

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

def pipeline_sentiment_analysis(model_name, df):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f"Using device: {device}")
    device_index = 0 if device == 'cuda' else -1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device_index)

    aspects = ["food", "place", "price", "service"]

    updated_data = extract_aspect_sentiments(df.to_dict(orient='records'), aspects, classifier)
    # df_updated = pd.DataFrame(updated_data)

    return updated_data


def pipeline_summarization(df_sentiment, model='Qwen/Qwen2.5-1.5B-Instruct'):
    result = {"category":[], "sentiment":[], "summary":[]}

    qwen_model = AutoModelForCausalLM.from_pretrained(model, device_map='auto')
    qwen_tokenizer = AutoTokenizer.from_pretrained(model)

    for category in df_sentiment['category'].unique():
    # Get each sentiment for the category
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            prompt = "Summarize the following reviews to give out the aspects that it talks most about:\n"
            
            for text in tqdm(df_sentiment[df_sentiment['sentiment_' + category] == sentiment]['text']):
                prompt += text + "\n"
                
            full_summary = ""
            
            inputs = qwen_tokenizer(prompt, return_tensors='pt').to('cuda' if torch.cuda.is_available() else 'cpu')
            
            with torch.no_grad():
                outputs = qwen_model.generate(**inputs, max_new_tokens=100)
            
            full_summary = qwen_tokenizer.decode(outputs[:, inputs['input_ids'].shape[-1]:][0], skip_special_tokens=True).strip() + " "

            result['category'].append(category)
            result['sentiment'].append(sentiment)
            result['summary'].append(full_summary)