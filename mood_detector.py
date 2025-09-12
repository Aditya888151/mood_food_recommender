# Mood Detection System using ML
# Supports: happy, sad, angry, stressed, relaxed, hungry, adventurous, energetic, comfort, light

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import re

# Setup NLTK for TextBlob
try:
    import nltk
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    nltk.download('punkt', quiet=True)
    nltk.download('brown', quiet=True)
except:
    pass

from textblob import TextBlob

class ImprovedMoodDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 3), min_df=1)
        self.model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
        self.is_trained = False
        
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def create_comprehensive_training_data(self):
        from app import parse_js_data
        
        # Load menu items for comprehensive training
        try:
            menu_items = parse_js_data('ItemData.js')
        except:
            menu_items = []
        
        training_texts = []
        training_moods = []
        
        # Basic mood examples
        mood_examples = {
            'happy': ["I'm feeling great", "What a wonderful day", "I'm so excited", "Life is beautiful", "I'm cheerful", "I feel amazing", "I'm happy", "feeling happy"],
            'sad': ["I'm feeling down", "I'm really sad", "Everything seems gloomy", "I'm heartbroken", "I'm depressed", "I feel empty", "I'm sad", "feeling sad"],
            'angry': ["I'm so angry", "This is frustrating", "I'm furious", "I'm really mad", "I'm irritated", "I'm livid", "I'm angry", "feeling angry"],
            'stressed': ["I'm so stressed", "I'm overwhelmed", "Too much pressure", "I'm anxious", "I'm worried", "I'm tense", "I'm stressed", "feeling stressed"],
            'relaxed': ["I'm feeling calm", "I'm so relaxed", "Everything is zen", "I'm at peace", "I'm chill", "I'm comfortable", "I'm relaxed", "feeling relaxed"],
            'adventurous': ["I want something new", "I'm feeling adventurous", "Let's explore", "Something different", "I'm adventurous", "feeling adventurous"],
            'energetic': ["I'm full of energy", "I'm pumped up", "I'm dynamic", "I'm charged up", "I'm energetic", "feeling energetic"],
            'comfort': ["I need comfort food", "Something cozy", "I need warmth", "Something familiar", "I want comfort", "need comfort"],
            'light': ["I want something light", "Something fresh", "Light and healthy", "Something simple", "I want light", "something light"]
        }
        
        # Add basic mood examples
        for mood, examples in mood_examples.items():
            training_texts.extend(examples)
            training_moods.extend([mood] * len(examples))
        
        # Generate comprehensive food examples (all as 'hungry')
        food_examples = []
        
        # Add specific menu items
        for item in menu_items[:100]:  # Limit to prevent too large dataset
            item_name = item['item_name'].lower()
            food_examples.extend([
                f"I want {item_name}",
                f"Give me {item_name}",
                f"I need {item_name}"
            ])
        
        # Add food categories
        categories = ['biryani', 'pizza', 'pasta', 'chicken', 'paneer', 'noodles', 'sandwich', 'dosa', 'momos', 'tikka', 'kabab', 'fish', 'egg', 'mutton', 'prawn', 'soup', 'salad', 'dessert', 'ice cream', 'coffee', 'tea', 'juice', 'cold drinks', 'lassi', 'curry', 'dal', 'rice']
        
        for category in categories:
            food_examples.extend([
                f"I want {category}",
                f"Give me {category}",
                f"I need {category}",
                f"I'm craving {category}",
                f"Show me {category}"
            ])
        
        # Add hunger-specific examples
        hunger_examples = ["I'm starving", "I'm so hungry", "I need food", "I'm famished", "I'm hungry", "feeling hungry", "getting hungry"]
        food_examples.extend(hunger_examples)
        
        # All food examples are 'hungry'
        training_texts.extend(food_examples)
        training_moods.extend(['hungry'] * len(food_examples))
        
        training_data = {
            'text': training_texts,
            'mood': training_moods
        }
        
        return pd.DataFrame(training_data)
    
    def train_model(self):
        # print("Creating training data...")
        df = self.create_comprehensive_training_data()
        
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['mood'], test_size=0.3, random_state=42, stratify=df['mood']
        )
        
        # print("Training model...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        self.model.fit(X_train_vec, y_train)
        
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained! Accuracy: {accuracy:.2f}")
        self.is_trained = True
        self.save_model()
        
    def save_model(self):
        with open('improved_mood_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('improved_mood_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print("Model saved!")
    
    def load_model(self):
        try:
            with open('improved_mood_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('improved_mood_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            self.is_trained = True
            print("Model loaded!")
            return True
        except FileNotFoundError:
            print("No saved model found. Please train first.")
            return False
    
    def predict_mood(self, text):
        if not self.is_trained:
            if not self.load_model():
                raise Exception("Model not trained.")
        
        processed_text = self.preprocess_text(text)
        text_vec = self.vectorizer.transform([processed_text])
        
        mood = self.model.predict(text_vec)[0]
        confidence = max(self.model.predict_proba(text_vec)[0])
        
        return mood, confidence
    
    def predict_mood_with_sentiment(self, text):
        mood, confidence = self.predict_mood(text)
        
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment
            
            if confidence < 0.4:
                if sentiment.polarity > 0.5:
                    mood = 'happy'
                    confidence = 0.7
                elif sentiment.polarity < -0.5:
                    mood = 'sad'
                    confidence = 0.7
                elif abs(sentiment.polarity) < 0.1:
                    mood = 'relaxed'
                    confidence = 0.6
        except:
            # Fallback to keyword-based sentiment if TextBlob fails
            text_lower = text.lower()
            positive_words = ['great', 'amazing', 'wonderful', 'fantastic', 'excellent', 'good']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate']
            
            if confidence < 0.4:
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)
                
                if positive_count > negative_count and positive_count > 0:
                    mood = 'happy'
                    confidence = 0.7
                elif negative_count > positive_count and negative_count > 0:
                    mood = 'sad'
                    confidence = 0.7
        
        return mood, confidence

