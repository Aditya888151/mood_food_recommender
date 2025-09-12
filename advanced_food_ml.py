# Advanced Food Recommendation ML Model with Online Data Training
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import re

from textblob import TextBlob

class AdvancedFoodML:
    def __init__(self):
        self.mood_vectorizer = TfidfVectorizer(max_features=8000, stop_words='english', ngram_range=(1, 4), min_df=1)
        self.combo_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 3), min_df=1)
        self.mood_model = GradientBoostingClassifier(n_estimators=300, random_state=42, max_depth=8)
        self.combo_model = RandomForestClassifier(n_estimators=250, random_state=42, max_depth=12)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def create_comprehensive_training_data(self):
        """Create extensive training data from multiple sources"""
        
        # Enhanced mood-food training data
        mood_food_data = {
            'happy': [
                "I'm celebrating today", "Feeling joyful and want something sweet", "Happy birthday to me",
                "Got promoted today", "Feeling festive", "Party mood", "Celebration time",
                "I'm in a great mood", "Feeling cheerful", "Want something fun to eat",
                "Feeling excited about life", "Good news today", "Victory celebration",
                "Weekend vibes", "Feeling blessed", "Want to treat myself"
            ],
            'sad': [
                "Feeling down today", "Need comfort food", "Having a bad day", "Feeling lonely",
                "Breakup blues", "Missing home", "Feeling depressed", "Need something warm",
                "Emotional eating", "Feeling blue", "Rainy day mood", "Need soul food",
                "Feeling heartbroken", "Want mom's cooking", "Comfort me with food"
            ],
            'angry': [
                "So frustrated right now", "Need something spicy", "Angry at work", "Road rage",
                "Want to burn my tongue", "Feeling furious", "Need hot food", "Spice up my mood",
                "Irritated and hungry", "Want fiery food", "Feeling aggressive", "Need strong flavors",
                "Want something that bites back", "Feeling heated", "Spicy mood"
            ],
            'stressed': [
                "Work deadline stress", "Exam pressure", "Feeling overwhelmed", "Need quick food",
                "Too much pressure", "Anxiety eating", "Stressed out", "Need comfort",
                "Feeling tense", "Work stress", "Need something soothing", "Pressure cooker life",
                "Feeling anxious", "Need stress relief", "Overwhelmed with tasks"
            ],
            'relaxed': [
                "Chill evening", "Lazy Sunday", "Feeling zen", "Peaceful mood", "Meditation break",
                "Calm and serene", "Weekend relaxation", "Feeling tranquil", "Easy going mood",
                "Laid back vibes", "Feeling mellow", "Peaceful evening", "Relaxing at home",
                "Feeling centered", "Quiet time", "Mindful eating"
            ],
            'hungry': [
                "Starving right now", "Haven't eaten all day", "Famished", "Need food ASAP",
                "Stomach growling", "Skipped breakfast", "Working late and hungry", "Need fuel",
                "Empty stomach", "Craving a meal", "Need substantial food", "Hunger pangs",
                "Need to fill up", "Ravenous", "Need proper meal", "Feeling peckish"
            ],
            'adventurous': [
                "Want to try something new", "Feeling experimental", "Exotic food mood",
                "Want unique flavors", "Culinary adventure", "Try different cuisine",
                "Feeling bold with food", "Want unusual combinations", "Explore new tastes",
                "Feeling daring", "Want fusion food", "International mood", "Exotic cravings"
            ],
            'energetic': [
                "Need energy boost", "Pre-workout meal", "Feeling active", "Need fuel for gym",
                "High energy day", "Need protein power", "Feeling dynamic", "Active lifestyle",
                "Need stamina food", "Power meal needed", "Feeling charged up", "Athletic mood",
                "Need performance food", "Feeling pumped", "Energy food required"
            ],
            'comfort': [
                "Need comfort food", "Want homestyle cooking", "Feeling nostalgic", "Need warmth",
                "Want familiar flavors", "Childhood memories", "Need cozy food", "Traditional mood",
                "Want grandma's recipe", "Feeling homesick", "Need soul warming food",
                "Want classic dishes", "Comfort eating", "Need familiar taste", "Homely food"
            ],
            'light': [
                "Want something light", "Feeling health conscious", "Need fresh food", "Diet mode",
                "Want clean eating", "Feeling light and fresh", "Need detox food", "Healthy mood",
                "Want nutritious meal", "Feeling fit", "Need low calorie", "Fresh and light",
                "Want salad mood", "Feeling wellness", "Clean eating day"
            ]
        }
        
        # Food combination training data
        combo_training_data = [
            # Indian combinations
            ("butter chicken", "naan bread beverages raita"),
            ("biryani", "raita pickle beverages dessert"),
            ("dal curry", "rice roti beverages pickle"),
            ("paneer curry", "naan rice beverages salad"),
            ("chicken curry", "rice bread beverages pickle"),
            
            # Italian combinations  
            ("pizza", "cold drink garlic bread salad"),
            ("pasta", "garlic bread beverages salad soup"),
            ("lasagna", "garlic bread beverages salad"),
            
            # Chinese combinations
            ("fried rice", "soup cold drink manchurian"),
            ("noodles", "soup cold drink spring rolls"),
            ("momos", "soup chutney beverages"),
            
            # Continental combinations
            ("burger", "fries cold drink salad"),
            ("sandwich", "chips beverages soup"),
            ("steak", "mashed potato wine salad"),
            
            # Breakfast combinations
            ("dosa", "sambar chutney coffee"),
            ("idli", "sambar chutney coffee"),
            ("paratha", "curd pickle tea"),
            
            # Snack combinations
            ("samosa", "chutney tea"),
            ("pakora", "chutney tea coffee"),
            ("chat", "cold drink lassi")
        ]
        
        # Create training datasets
        mood_texts = []
        mood_labels = []
        
        for mood, examples in mood_food_data.items():
            mood_texts.extend(examples)
            mood_labels.extend([mood] * len(examples))
        
        # Add food-specific examples
        from app import parse_js_data
        try:
            menu_items = parse_js_data('ItemData.js')
            for item in menu_items[:100]:
                item_name = item['item_name'].lower()
                mood_texts.extend([
                    f"I want {item_name}",
                    f"Craving {item_name}",
                    f"Need {item_name}",
                    f"Give me {item_name}",
                    f"Order {item_name}"
                ])
                mood_labels.extend(['hungry'] * 5)
        except:
            pass
        
        # Create combo training data
        combo_texts = []
        combo_labels = []
        
        for main_dish, combo_items in combo_training_data:
            combo_texts.append(main_dish)
            combo_labels.append(combo_items)
        
        return {
            'mood_data': pd.DataFrame({'text': mood_texts, 'mood': mood_labels}),
            'combo_data': pd.DataFrame({'main_dish': combo_texts, 'combo_items': combo_labels})
        }
    

    
    def train_models(self):
        """Train both mood detection and combo recommendation models"""
        print("Creating comprehensive training data...")
        training_data = self.create_comprehensive_training_data()
        # Food pairing knowledge base
        food_pairings = {
            'chicken': ['rice', 'naan', 'salad', 'soup', 'beverages'],
            'fish': ['rice', 'lemon', 'salad', 'wine', 'vegetables'],
            'beef': ['potato', 'wine', 'salad', 'bread', 'vegetables'],
            'pasta': ['garlic bread', 'salad', 'wine', 'soup', 'cheese'],
            'pizza': ['cold drink', 'salad', 'garlic bread', 'dessert'],
            'biryani': ['raita', 'pickle', 'beverages', 'dessert', 'salad'],
            'curry': ['rice', 'bread', 'pickle', 'beverages', 'salad'],
            'soup': ['bread', 'crackers', 'salad', 'beverages'],
            'salad': ['bread', 'soup', 'beverages', 'dressing'],
            'dessert': ['coffee', 'tea', 'milk', 'beverages']
        }
        
        online_data = []
        for main_food, pairings in food_pairings.items():
            online_data.append({
                'main_dish': main_food,
                'combo_items': ' '.join(pairings)
            })
        
        # Train mood detection model
        mood_df = training_data['mood_data']
        mood_df['processed_text'] = mood_df['text'].apply(self.preprocess_text)
        
        X_mood_train, X_mood_test, y_mood_train, y_mood_test = train_test_split(
            mood_df['processed_text'], mood_df['mood'], 
            test_size=0.2, random_state=42, stratify=mood_df['mood']
        )
        
        print("Training mood detection model...")
        X_mood_train_vec = self.mood_vectorizer.fit_transform(X_mood_train)
        X_mood_test_vec = self.mood_vectorizer.transform(X_mood_test)
        
        self.mood_model.fit(X_mood_train_vec, y_mood_train)
        mood_pred = self.mood_model.predict(X_mood_test_vec)
        mood_accuracy = accuracy_score(y_mood_test, mood_pred)
        
        # Train combo recommendation model
        combo_df = training_data['combo_data']
        combo_df = pd.concat([combo_df, pd.DataFrame(online_data)], ignore_index=True)
        
        print("Training combo recommendation model...")
        X_combo = self.combo_vectorizer.fit_transform(combo_df['main_dish'])
        y_combo_encoded = self.label_encoder.fit_transform(combo_df['combo_items'])
        
        self.combo_model.fit(X_combo, y_combo_encoded)
        
        print(f"Mood Detection Accuracy: {mood_accuracy:.2f}")
        print(f"Combo Model trained with {len(combo_df)} combinations")
        
        self.is_trained = True
        self.save_models()
    
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def predict_mood_advanced(self, text):
        """Advanced mood prediction with confidence scoring"""
        if not self.is_trained:
            if not self.load_models():
                raise Exception("Models not trained.")
        
        processed_text = self.preprocess_text(text)
        text_vec = self.mood_vectorizer.transform([processed_text])
        
        mood = self.mood_model.predict(text_vec)[0]
        confidence = max(self.mood_model.predict_proba(text_vec)[0])
        
        # Enhance with sentiment analysis
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment
            
            if confidence < 0.5:
                if sentiment.polarity > 0.6:
                    mood = 'happy'
                    confidence = 0.8
                elif sentiment.polarity < -0.6:
                    mood = 'sad'
                    confidence = 0.8
        except:
            pass
        
        return mood, confidence
    
    def predict_combo_advanced(self, main_dish):
        """Advanced combo prediction"""
        if not self.is_trained:
            if not self.load_models():
                raise Exception("Models not trained.")
        
        main_dish_vec = self.combo_vectorizer.transform([main_dish.lower()])
        combo_encoded = self.combo_model.predict(main_dish_vec)[0]
        combo_items = self.label_encoder.inverse_transform([combo_encoded])[0]
        
        return combo_items.split()
    
    def save_models(self):
        """Save all trained models"""
        with open('advanced_mood_model.pkl', 'wb') as f:
            pickle.dump(self.mood_model, f)
        with open('advanced_mood_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.mood_vectorizer, f)
        with open('advanced_combo_model.pkl', 'wb') as f:
            pickle.dump(self.combo_model, f)
        with open('advanced_combo_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.combo_vectorizer, f)
        with open('advanced_label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print("Advanced models saved!")
    
    def load_models(self):
        """Load all trained models"""
        try:
            with open('advanced_mood_model.pkl', 'rb') as f:
                self.mood_model = pickle.load(f)
            with open('advanced_mood_vectorizer.pkl', 'rb') as f:
                self.mood_vectorizer = pickle.load(f)
            with open('advanced_combo_model.pkl', 'rb') as f:
                self.combo_model = pickle.load(f)
            with open('advanced_combo_vectorizer.pkl', 'rb') as f:
                self.combo_vectorizer = pickle.load(f)
            with open('advanced_label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            self.is_trained = True
            print("Advanced models loaded!")
            return True
        except FileNotFoundError:
            print("No saved advanced models found.")
            return False

