import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import re
from textblob import TextBlob

class MoodDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = LogisticRegression(random_state=42)
        self.is_trained = False
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def create_training_data(self):
        """Create training dataset with mood-related text samples"""
        training_data = {
            'text': [
                # Happy mood texts
                "I'm feeling great today", "What a wonderful day", "I'm so excited", 
                "Life is beautiful", "I'm in a fantastic mood", "Everything is perfect",
                "I'm cheerful and energetic", "Feeling blessed and joyful", "I'm on cloud nine",
                "Today is amazing", "I'm feeling optimistic", "I'm thrilled about everything",
                
                # Sad mood texts
                "I'm feeling down", "I'm really sad today", "Everything seems gloomy",
                "I'm heartbroken", "Feeling blue and lonely", "I'm depressed",
                "Nothing is going right", "I feel empty inside", "I'm crying",
                "Life is tough", "I'm feeling melancholy", "I'm devastated",
                
                # Angry mood texts
                "I'm so angry right now", "This is frustrating", "I'm furious",
                "I'm really mad", "This makes me rage", "I'm irritated",
                "I'm boiling with anger", "I'm outraged", "This is infuriating",
                "I'm livid", "I'm steaming mad", "I'm really pissed off",
                
                # Stressed mood texts
                "I'm so stressed out", "I'm overwhelmed", "Too much pressure",
                "I'm anxious about everything", "I can't handle this", "I'm burnt out",
                "I'm worried sick", "I'm under so much stress", "I'm panicking",
                "I'm tense and nervous", "I'm feeling pressured", "I'm exhausted from stress",
                
                # Relaxed mood texts
                "I'm feeling calm and peaceful", "I'm so relaxed", "Everything is zen",
                "I'm at peace", "Feeling tranquil", "I'm completely chill",
                "I'm serene and content", "Feeling laid back", "I'm in my comfort zone",
                "I'm feeling mellow", "Everything is smooth", "I'm totally at ease",
                
                # Hungry mood texts
                "I'm starving", "I'm so hungry", "I need food now",
                "My stomach is growling", "I'm famished", "I could eat anything",
                "I'm craving food", "I need to eat something", "I'm really hungry",
                "Food sounds amazing right now", "I'm dying for some food", "I need a meal",
                
                # Adventurous mood texts
                "I want to try something new", "I'm feeling adventurous", "Let's explore",
                "I want something exotic", "I'm ready for an adventure", "Something different please",
                "I'm feeling bold today", "Let's try something unique", "I want to experiment",
                "I'm in the mood for something unusual", "Let's be adventurous", "Something exciting",
                
                # Energetic mood texts
                "I'm full of energy", "I'm pumped up", "I'm feeling dynamic",
                "I'm charged up", "I'm bursting with energy", "I'm feeling active",
                "I'm ready to conquer the world", "I'm feeling vibrant", "I'm energized",
                "I'm feeling powerful", "I'm hyped up", "I'm feeling electric",
                
                # Comfort mood texts
                "I need comfort food", "I want something cozy", "I need warmth",
                "I want something familiar", "I need comfort", "Something homely please",
                "I want something soothing", "I need emotional comfort", "Something nurturing",
                "I want something that feels like home", "I need comfort and warmth", "Something comforting",
                
                # Light mood texts
                "I want something light", "I don't want heavy food", "Something fresh please",
                "I want to eat light", "Something not too filling", "I prefer light meals",
                "Something refreshing", "I want something clean", "Light and healthy please",
                "Something easy on the stomach", "I want something simple", "Light and fresh"
            ],
            'mood': [
                # Happy (12 samples)
                'happy', 'happy', 'happy', 'happy', 'happy', 'happy',
                'happy', 'happy', 'happy', 'happy', 'happy', 'happy',
                
                # Sad (12 samples)
                'sad', 'sad', 'sad', 'sad', 'sad', 'sad',
                'sad', 'sad', 'sad', 'sad', 'sad', 'sad',
                
                # Angry (12 samples)
                'angry', 'angry', 'angry', 'angry', 'angry', 'angry',
                'angry', 'angry', 'angry', 'angry', 'angry', 'angry',
                
                # Stressed (12 samples)
                'stressed', 'stressed', 'stressed', 'stressed', 'stressed', 'stressed',
                'stressed', 'stressed', 'stressed', 'stressed', 'stressed', 'stressed',
                
                # Relaxed (12 samples)
                'relaxed', 'relaxed', 'relaxed', 'relaxed', 'relaxed', 'relaxed',
                'relaxed', 'relaxed', 'relaxed', 'relaxed', 'relaxed', 'relaxed',
                
                # Hungry (12 samples)
                'hungry', 'hungry', 'hungry', 'hungry', 'hungry', 'hungry',
                'hungry', 'hungry', 'hungry', 'hungry', 'hungry', 'hungry',
                
                # Adventurous (12 samples)
                'adventurous', 'adventurous', 'adventurous', 'adventurous', 'adventurous', 'adventurous',
                'adventurous', 'adventurous', 'adventurous', 'adventurous', 'adventurous', 'adventurous',
                
                # Energetic (12 samples)
                'energetic', 'energetic', 'energetic', 'energetic', 'energetic', 'energetic',
                'energetic', 'energetic', 'energetic', 'energetic', 'energetic', 'energetic',
                
                # Comfort (12 samples)
                'comfort', 'comfort', 'comfort', 'comfort', 'comfort', 'comfort',
                'comfort', 'comfort', 'comfort', 'comfort', 'comfort', 'comfort',
                
                # Light (12 samples)
                'light', 'light', 'light', 'light', 'light', 'light',
                'light', 'light', 'light', 'light', 'light', 'light'
            ]
        }
        return pd.DataFrame(training_data)
    
    def train_model(self):
        """Train the mood detection model"""
        print("Creating training data...")
        df = self.create_training_data()
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['mood'], test_size=0.2, random_state=42, stratify=df['mood']
        )
        
        print("Training model...")
        # Vectorize text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        
        # Save model and vectorizer
        self.save_model()
        
    def save_model(self):
        """Save trained model and vectorizer"""
        with open('mood_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('mood_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print("Model saved successfully!")
    
    def load_model(self):
        """Load trained model and vectorizer"""
        try:
            with open('mood_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('mood_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            self.is_trained = True
            print("Model loaded successfully!")
            return True
        except FileNotFoundError:
            print("No saved model found. Please train the model first.")
            return False
    
    def predict_mood(self, text):
        """Predict mood from text"""
        if not self.is_trained:
            if not self.load_model():
                raise Exception("Model not trained. Please train the model first.")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Vectorize
        text_vec = self.vectorizer.transform([processed_text])
        
        # Predict
        mood = self.model.predict(text_vec)[0]
        confidence = max(self.model.predict_proba(text_vec)[0])
        
        return mood, confidence
    
    def predict_mood_with_sentiment(self, text):
        """Enhanced mood prediction using sentiment analysis"""
        # Get basic mood prediction
        mood, confidence = self.predict_mood(text)
        
        # Add sentiment analysis for better accuracy
        blob = TextBlob(text)
        sentiment = blob.sentiment
        
        # Adjust mood based on sentiment if confidence is low
        if confidence < 0.6:
            if sentiment.polarity > 0.3:
                mood = 'happy'
            elif sentiment.polarity < -0.3:
                mood = 'sad'
            elif abs(sentiment.polarity) < 0.1 and sentiment.subjectivity < 0.3:
                mood = 'relaxed'
        
        return mood, confidence

if __name__ == "__main__":
    # Create and train the model
    detector = MoodDetector()
    detector.train_model()
    
    # Test the model
    test_texts = [
        "I'm feeling really happy today!",
        "I'm so stressed about work",
        "I'm starving and need food",
        "I want to try something new and exciting"
    ]
    
    print("\nTesting the model:")
    for text in test_texts:
        mood, confidence = detector.predict_mood_with_sentiment(text)
        print(f"Text: '{text}' -> Mood: {mood} (Confidence: {confidence:.2f})")