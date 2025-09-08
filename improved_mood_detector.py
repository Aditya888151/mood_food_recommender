# Mood Detection System using ML
# Supports: happy, sad, angry, stressed, relaxed, hungry, adventurous, energetic, comfort, light

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import re
from textblob import TextBlob

class ImprovedMoodDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def create_comprehensive_training_data(self):
        training_data = {
            'text': [
                # Happy examples
                "I'm feeling great today", "What a wonderful day", "I'm so excited", "Life is beautiful",
                "I'm in a fantastic mood", "Everything is perfect", "I'm cheerful", "I feel amazing",
                "I'm thrilled", "I'm overjoyed", "I'm ecstatic", "I feel wonderful",
                
                # Sad examples  
                "I'm feeling down", "I'm really sad today", "Everything seems gloomy", "I'm heartbroken",
                "Feeling blue", "I'm depressed", "Nothing is going right", "I feel empty",
                "I'm crying", "Life is tough", "I feel hopeless", "I'm devastated",
                
                # Angry examples
                "I'm so angry right now", "This is frustrating", "I'm furious", "I'm really mad",
                "This makes me rage", "I'm irritated", "I'm boiling with anger", "I'm outraged",
                "I'm livid", "I'm steaming mad", "I'm enraged", "I'm seething",
                
                # Stressed examples
                "I'm so stressed out", "I'm overwhelmed", "Too much pressure", "I'm anxious",
                "I can't handle this", "I'm burnt out", "I'm worried sick", "I'm panicking",
                "I'm tense", "I'm frazzled", "I'm at breaking point", "I'm on edge",
                
                # Relaxed examples
                "I'm feeling calm", "I'm so relaxed", "Everything is zen", "I'm at peace",
                "Feeling tranquil", "I'm completely chill", "I'm serene", "I'm mellow",
                "I'm totally at ease", "I'm feeling centered", "I'm carefree", "I'm comfortable",
                
                # Hungry examples
                "I'm starving", "I'm so hungry", "I need food now", "My stomach is growling",
                "I'm famished", "I could eat anything", "I'm craving food", "I need to eat",
                "I'm ravenous", "I could eat a horse", "I'm feeling peckish", "I need sustenance",
                
                # Adventurous examples
                "I want to try something new", "I'm feeling adventurous", "Let's explore", "I want something exotic",
                "I'm ready for an adventure", "Something different please", "I'm feeling bold", "Let's try something unique",
                "I want to experiment", "Something exciting", "I want to discover new flavors", "I'm ready to take risks",
                
                # Energetic examples
                "I'm full of energy", "I'm pumped up", "I'm feeling dynamic", "I'm charged up",
                "I'm bursting with energy", "I'm feeling active", "I'm ready to conquer", "I'm feeling vibrant",
                "I'm energized", "I'm hyped up", "I'm buzzing with energy", "I'm supercharged",
                
                # Comfort examples
                "I need comfort food", "I want something cozy", "I need warmth", "I want something familiar",
                "I need comfort", "Something homely please", "I want something soothing", "I need emotional comfort",
                "Something nurturing", "I want something that feels like home", "I need comfort and warmth", "Something comforting",
                
                # Light examples
                "I want something light", "I don't want heavy food", "Something fresh please", "I want to eat light",
                "Something not too filling", "I prefer light meals", "Something refreshing", "I want something clean",
                "Light and healthy please", "Something easy on the stomach", "I want something simple", "Light and fresh"
            ],
            'mood': (
                ['happy'] * 12 + ['sad'] * 12 + ['angry'] * 12 + ['stressed'] * 12 + 
                ['relaxed'] * 12 + ['hungry'] * 12 + ['adventurous'] * 12 + 
                ['energetic'] * 12 + ['comfort'] * 12 + ['light'] * 12
            )
        }
        return pd.DataFrame(training_data)
    
    def train_model(self):
        print("Creating training data...")
        df = self.create_comprehensive_training_data()
        
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['mood'], test_size=0.2, random_state=42, stratify=df['mood']
        )
        
        print("Training model...")
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
            pass
        
        return mood, confidence

if __name__ == "__main__":
    detector = ImprovedMoodDetector()
    detector.train_model()
    
    # Test examples
    test_texts = [
        "I'm feeling really happy today!",
        "I'm so stressed about work",
        "I'm starving and need food",
        "I want to try something adventurous"
    ]
    
    print("\nTesting:")
    for text in test_texts:
        mood, confidence = detector.predict_mood_with_sentiment(text)
        print(f"'{text}' -> {mood} ({confidence:.2f})")