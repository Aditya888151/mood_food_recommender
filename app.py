# Mood-based Food Recommendation API - Production Version
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import re
import logging
import os
import random
from datetime import datetime

# Setup NLTK for TextBlob (Railway fix)
try:
    import nltk
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Download required NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('brown', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print("NLTK data setup completed")
except Exception as e:
    print(f"NLTK setup warning: {e} - continuing with fallbacks")

from textblob import TextBlob

# Setup production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import ML detectors, fallback to simple detection
try:
    from advanced_food_ml import AdvancedFoodML
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False

try:
    from mood_detector import ImprovedMoodDetector
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Simple mood detection (always available as fallback)
mood_keywords = {
    'happy': ['happy', 'excited', 'great', 'wonderful', 'fantastic', 'amazing'],
    'sad': ['sad', 'down', 'depressed', 'blue', 'heartbroken', 'crying'],
    'angry': ['angry', 'mad', 'furious', 'rage', 'irritated', 'frustrated', 'spicy', 'hot', 'fiery'],
    'stressed': ['stressed', 'overwhelmed', 'pressure', 'anxious', 'worried'],
    'relaxed': ['relaxed', 'calm', 'peaceful', 'zen', 'tranquil', 'chill'],
    'hungry': ['hungry', 'starving', 'famished', 'craving', 'need food', 'eat', 'food'],
    'adventurous': ['adventurous', 'explore', 'new', 'different', 'unique'],
    'energetic': ['energetic', 'pumped', 'dynamic', 'charged', 'active'],
    'comfort': ['comfort', 'cozy', 'warm', 'familiar', 'homely'],
    'light': ['light', 'fresh', 'healthy', 'clean', 'simple']
}

def detect_food_intent(text):
    """Detect if text is primarily about specific food items"""
    text_lower = text.lower()
    
    # Food-specific keywords that indicate direct food requests
    food_request_patterns = {
        'biryani': ['biryani', 'biriyani', 'pulao'],
        'pizza': ['pizza'],
        'pasta': ['pasta'],
        'chicken': ['chicken', 'murgh', 'meat'],
        'paneer': ['paneer'],
        'noodles': ['noodles','maggi', 'hakka', 'schezwan'],
        'momos': ['momos', 'momo'],
        'sandwich': ['sandwich'],
        'dosa': ['dosa'],
        'soup': ['soup'],
        'rice': ['rice', 'chawal'],
        'curry': ['curry', 'masala'],
        'tikka': ['tikka'],
        'kabab': ['kabab', 'kebab'],
        'fish': ['fish', 'seafood'],
        'egg': ['egg', 'omelette'],
        'mutton': ['mutton', 'lamb', 'gosh'],
        'prawn': ['prawn', 'shrimp'],
        'cold drinks': ['cold drink', 'cold drinks', 'soft drink', 'soda', 'pepsi', 'coke', 'drink'],
        'juice': ['juice', 'fresh juice'],
        'coffee': ['coffee'],
        'tea': ['tea', 'chai'],
        'lassi': ['lassi'],
        'shake': ['shake', 'milkshake'],
        'smoothie': ['smoothie'], 
        'thali': ['thali', 'full meal ', 'meal']
    }
    
    # Direct food request indicators
    request_words = ['want', 'need', 'give me', 'i want', 'only', 'just']
    
    detected_foods = []
    for food_type, keywords in food_request_patterns.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_foods.append(food_type)
    
    has_request_word = any(word in text_lower for word in request_words)
    
    # If specific food mentioned with request words, it's a food intent
    if detected_foods and has_request_word:
        return True, detected_foods
    
    # If only food mentioned without mood words, it's likely food intent
    mood_words = ['happy', 'sad', 'angry', 'stressed','not well','well', 'good', 'simple', 'relaxed', 'hungry', 'feeling', 'mood']
    has_mood_words = any(word in text_lower for word in mood_words)
    
    if detected_foods and not has_mood_words:
        return True, detected_foods
    
    return False, detected_foods

def simple_mood_detect(text):
    text_lower = text.lower()
    
    # First check if this is a direct food request
    is_food_intent, detected_foods = detect_food_intent(text)
    if is_food_intent:
        return 'hungry', 0.9  # High confidence for direct food requests
    
    # Priority detection for hunger-related phrases
    hunger_phrases = ['feeling hungry', 'am hungry', 'getting hungry', 'want food', 'need food', 'craving food', 'craving', 'hungry', 'hunger', "i'm feeling hungry", "i'm hungry"]
    if any(phrase in text_lower for phrase in hunger_phrases):
        return 'hungry', 0.8
    
    # Check for food mentions that indicate hunger
    food_mentions = ['biryani', 'pizza', 'pasta', 'noodles', 'chicken', 'rice', 'curry', 'sandwich', 'thali', 'meal']
    hunger_indicators = ['want', 'need', 'craving', 'wants', 'needs']
    
    has_food = any(food in text_lower for food in food_mentions)
    has_hunger_indicator = any(indicator in text_lower for indicator in hunger_indicators)
    
    if has_food and has_hunger_indicator:
        return 'hungry', 0.7
    
    # Regular keyword matching
    mood_scores = {}
    for mood, keywords in mood_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            mood_scores[mood] = score
    
    if mood_scores:
        best_mood = max(mood_scores, key=mood_scores.get)
        confidence = min(0.9, 0.5 + mood_scores[best_mood] * 0.1)
        return best_mood, confidence
    
    # Sentiment analysis fallback
    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        if sentiment > 0.3:
            return 'happy', 0.7
        elif sentiment < -0.3:
            return 'sad', 0.7
    except:
        pass
    
    return 'relaxed', 0.5

# Initialize advanced ML models
if ADVANCED_ML_AVAILABLE:
    advanced_ml = AdvancedFoodML()
    try:
        advanced_ml.load_models()
        print("Advanced ML models loaded successfully")
    except:
        print("Training advanced ML models...")
        advanced_ml.train_models()
elif ML_AVAILABLE:
    mood_detector = ImprovedMoodDetector()
    try:
        mood_detector.load_model()
        print("ML mood detector loaded successfully")
    except:
        print("Training mood detection model...")
        mood_detector.train_model()
else:
    print("Using simple keyword-based mood detection")

def parse_js_data(file_path):
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    item_pattern = r'\{\s*item_name:\s*["\']([^"\']*)["\'],[^}]*category:\s*["\']([^"\']*)["\'],[^}]*qty:\s*(\d+),[^}]*nc:\s*(true|false),[^}]*ttp:\s*(\d+)[^}]*\}'
    matches = re.findall(item_pattern, content, re.DOTALL)
    
    for match in matches:
        item = {
            "item_name": match[0],
            "category": match[1],
            "qty": int(match[2]),
            "nc": match[3] == 'true',
            "ttp": int(match[4])
        }
        items.append(item)
    
    return items

# Load menu items
menu_items = parse_js_data('ItemData.js')
print(f"Loaded {len(menu_items)} menu items")

# All available categories for dynamic selection
all_categories = [
    "Rice / Biryani", "Indian Main Course Veg", "Indian Main Course Non Veg", 
    "Pizza", "Pasta", "Chinese Veg", "Chinese Non Veg", "Smokey Tandoori Roasted Non Veg",
    "Smokey Tandoori Starter Veg", "Soup", "Salad", "Beverages", "Dessert", "Ice Cream",
    "Sandwich", "Steam Momos", "Hi Tea", "Indian Breads", "Raita", "Japanese",
    "Continental Sizzler", "Fish / Prawns", "Continental Starter", "Bar Bite",
    "Fried Rice / Noodles", "South Indian", "Wrap", "Kathi Roll"
]

# Enhanced mood to food category mapping with better hungry options
mood_to_category = {
    "happy": ["Beverages", "Dessert", "Ice Cream", "Hi Tea", "Pizza"],
    "sad": ["Soup", "Indian Main Course Veg", "Indian Main Course Non Veg", "Dessert", "Ice Cream"],
    "angry": ["Chinese Non Veg", "Smokey Tandoori Roasted Non Veg", "Bar Bite"],
    "stressed": ["Beverages", "Hi Tea", "Soup", "Dessert"],
    "relaxed": ["Salad", "South Indian", "Japanese", "Raita", "Beverages"],
    "hungry": ["Rice / Biryani", "Indian Main Course Non Veg", "Indian Main Course Veg", "Pizza", "Pasta", "Chinese Non Veg", "Smokey Tandoori Roasted Non Veg", "Sandwich"],
    "adventurous": ["Japanese", "Continental Sizzler", "Fish / Prawns", "Continental Starter"],
    "energetic": ["Chinese Veg", "Chinese Non Veg", "Fried Rice / Noodles", "Steam Momos"],
    "comfort": ["Indian Main Course Veg", "Indian Main Course Non Veg", "Soup", "Indian Breads"],
    "light": ["Salad", "Soup", "South Indian", "Raita", "Hi Tea"]
}

def get_dynamic_categories(mood, diet_type):
    """Get dynamic categories that change every time"""
    import time
    random.seed(int(time.time() * 1000))
    
    # Start with mood-based categories
    base_categories = mood_to_category.get(mood, ["Pizza", "Pasta", "Rice / Biryani"])
    
    # Add random categories from all available
    extra_categories = [cat for cat in all_categories if cat not in base_categories]
    random.shuffle(extra_categories)
    
    # Combine and shuffle for complete randomness
    dynamic_categories = base_categories + extra_categories[:5]
    random.shuffle(dynamic_categories)
    
    return dynamic_categories[:8]  # Return 8 categories max

@app.route('/')
def home():
    return "Mood-based Food Recommendation API is running!"

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON data required"}), 400
        
        if 'text' in data:
            text = data.get('text', '').strip()
            if not text or len(text) > 500:
                return jsonify({"error": "Text is required and must be under 500 characters"}), 400
            
            try:
                # First check if this is a direct food request (prioritize over ML)
                is_food_intent, detected_foods = detect_food_intent(text)
                if is_food_intent:
                    detected_mood, confidence = 'hungry', 0.9
                elif ADVANCED_ML_AVAILABLE:
                    detected_mood, confidence = advanced_ml.predict_mood_advanced(text)
                elif ML_AVAILABLE:
                    detected_mood, confidence = mood_detector.predict_mood_with_sentiment(text)
                else:
                    detected_mood, confidence = simple_mood_detect(text)
                mood = detected_mood.lower()
            except Exception as e:
                print(f"Mood detection error: {e}")
                # Fallback to simple detection
                detected_mood, confidence = simple_mood_detect(text)
                mood = detected_mood.lower()
        else:
            mood = data.get("mood", "").lower()
            confidence = 1.0
            
        if not mood:
            return jsonify({"error": "Mood is required"}), 400

        diet_type = data.get("diet_type", "both").lower()
        
        # Get dynamic categories that change every time
        dynamic_categories = get_dynamic_categories(mood, diet_type)
        
        # Get smart recommendations with dynamic categories
        recommendations = get_smart_recommendations_with_categories(mood, diet_type, dynamic_categories, text if 'text' in data else '')
        
        # Use dynamic categories
        categories = dynamic_categories
        
        return jsonify({
            "detected_mood": mood,
            "confidence": round(confidence, 2),
            "categories": categories,
            "diet_type": diet_type,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations)
        })
    
    except Exception as e:
        logger.error(f"Recommend endpoint error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/categories', methods=['GET'])
def list_categories():
    categories = list(set(item["category"] for item in menu_items))
    return jsonify(sorted(categories))

@app.route('/moods', methods=['GET'])
def list_moods():
    return jsonify(sorted(mood_to_category.keys()))

@app.route('/combo', methods=['POST'])
def combo_recommend():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON data required"}), 400
        
        if 'text' in data:
            text = data.get('text', '').strip()
            if not text or len(text) > 500:
                return jsonify({"error": "Text is required and must be under 500 characters"}), 400
            
            try:
                is_food_intent, detected_foods = detect_food_intent(text)
                if is_food_intent:
                    detected_mood, confidence = 'hungry', 0.9
                elif ML_AVAILABLE:
                    detected_mood, confidence = mood_detector.predict_mood_with_sentiment(text)
                else:
                    detected_mood, confidence = simple_mood_detect(text)
                mood = detected_mood.lower()
            except Exception as e:
                detected_mood, confidence = simple_mood_detect(text)
                mood = detected_mood.lower()
        else:
            mood = data.get("mood", "").lower()
            confidence = 1.0
            
        if not mood:
            return jsonify({"error": "Mood is required"}), 400

        diet_type = data.get("diet_type", "both").lower()
        
        # Get single main item with dynamic category selection
        dynamic_categories = get_dynamic_categories(mood, diet_type)
        main_item = get_single_main_item_dynamic(text if 'text' in data else '', mood, diet_type, dynamic_categories)
        if not main_item:
            # Fallback to any item matching diet preference
            if diet_type == 'veg':
                main_item = next((item for item in menu_items if not item['nc']), menu_items[0])
            elif diet_type == 'non-veg':
                main_item = next((item for item in menu_items if item['nc']), menu_items[0])
            else:
                main_item = menu_items[0]
        
        # Get combo items
        combo_items = get_combo_items(main_item, diet_type)
        
        return jsonify({
            "detected_mood": mood,
            "confidence": round(confidence, 2),
            "diet_type": diet_type,
            "main_item": main_item,
            "combo_items": combo_items,
            "total_combo_items": len(combo_items)
        })
    
    except Exception as e:
        logger.error(f"Combo endpoint error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

def get_single_main_item_dynamic(text, mood, diet_type, categories):
    """Get single main item using dynamic categories"""
    import time
    random.seed(int(time.time() * 1000))
    
    text_lower = text.lower() if text else ''
    
    # Check for specific food requests first
    food_keywords = {
        'biryani': ['biryani', 'biriyani', 'pulao'],
        'pizza': ['pizza'],
        'pasta': ['pasta', 'penne'],
        'chicken': ['chicken', 'murgh'],
        'paneer': ['paneer'],
        'noodles': ['noodles', 'hakka', 'schezwan']
    }
    
    for food_type, keywords in food_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            matching_items = []
            for item in menu_items:
                item_name_lower = item['item_name'].lower()
                if any(keyword in item_name_lower for keyword in keywords):
                    if ((diet_type == 'veg' and not item['nc']) or 
                        (diet_type == 'non-veg' and item['nc']) or 
                        diet_type == 'both'):
                        matching_items.append(item)
            
            if matching_items:
                random.shuffle(matching_items)
                return matching_items[0]
    
    # Use dynamic categories for main item selection
    main_course_categories = [cat for cat in categories if cat in [
        'Rice / Biryani', 'Pizza', 'Pasta', 'Indian Main Course Non Veg', 
        'Indian Main Course Veg', 'Chinese Non Veg', 'Chinese Veg'
    ]]
    
    if not main_course_categories:
        main_course_categories = categories[:3]  # Use first 3 categories
    
    random.shuffle(main_course_categories)
    
    for category in main_course_categories:
        category_items = [item for item in menu_items if item['category'] == category]
        
        if diet_type == 'veg':
            category_items = [item for item in category_items if not item['nc']]
        elif diet_type == 'non-veg':
            category_items = [item for item in category_items if item['nc']]
        
        if category_items:
            random.shuffle(category_items)
            return category_items[0]
    
    # Fallback
    if diet_type == 'veg':
        fallback = [item for item in menu_items if not item['nc']]
    elif diet_type == 'non-veg':
        fallback = [item for item in menu_items if item['nc']]
    else:
        fallback = menu_items
    
    if fallback:
        random.shuffle(fallback)
        return fallback[0]
    
    return None

def get_single_main_item(text, mood, diet_type):
    """Get single best main item based on text and mood with variety"""
    text_lower = text.lower() if text else ''
    
    # Check for specific combinations first (e.g., "chicken biryani")
    if 'chicken biryani' in text_lower or 'chicken biriyani' in text_lower:
        chicken_biryani_items = [item for item in menu_items 
                               if 'chicken' in item['item_name'].lower() and 'biryani' in item['item_name'].lower()]
        if chicken_biryani_items:
            random.shuffle(chicken_biryani_items)
            return chicken_biryani_items[0]
    
    if 'mutton biryani' in text_lower or 'mutton biriyani' in text_lower:
        mutton_biryani_items = [item for item in menu_items 
                              if 'mutton' in item['item_name'].lower() and 'biryani' in item['item_name'].lower()]
        if mutton_biryani_items:
            random.shuffle(mutton_biryani_items)
            return mutton_biryani_items[0]
    
    # Food-specific selection - find exact matches
    food_keywords = {
        'biryani': ['biryani', 'biriyani', 'pulao'],
        'pizza': ['pizza'],
        'pasta': ['pasta', 'penne'],
        'chicken': ['chicken', 'murgh'],
        'paneer': ['paneer'],
        'noodles': ['noodles', 'hakka', 'schezwan'],
        'sandwich': ['sandwich'],
        'burger': ['burger'],
        'dosa': ['dosa'],
        'momos': ['momos', 'momo'],
        'tikka': ['tikka'],
        'kabab': ['kabab', 'kebab'],
        'fish': ['fish'],
        'egg': ['egg', 'omelette'],
        'mutton': ['mutton', 'lamb'],
        'prawn': ['prawn', 'shrimp']
    }
    
    # Check for specific food mentions and find exact matches
    for food_type, keywords in food_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            # Find items that match this specific food type
            matching_items = []
            
            # Prioritize main course categories for proper meals
            main_course_categories = ['Indian Main Course Veg', 'Indian Main Course Non Veg', 'Rice / Biryani', 'Pizza', 'Pasta', 'Chinese Veg', 'Chinese Non Veg']
            
            for item in menu_items:
                item_name_lower = item['item_name'].lower()
                if any(keyword in item_name_lower for keyword in keywords):
                    if ((diet_type == 'veg' and not item['nc']) or 
                        (diet_type == 'non-veg' and item['nc']) or 
                        diet_type == 'both'):
                        # Prioritize main course items
                        if item['category'] in main_course_categories:
                            matching_items.insert(0, item)  # Add to front
                        else:
                            matching_items.append(item)  # Add to back
            
            # Return random matching item for variety
            if matching_items:
                random.shuffle(matching_items)
                return matching_items[0]
    
    # Enhanced mood-based selection with better variety
    if mood == 'hungry':
        # For hungry mood, prioritize popular main dishes
        popular_categories = ['Rice / Biryani', 'Pizza', 'Pasta', 'Indian Main Course Non Veg', 'Indian Main Course Veg']
    else:
        popular_categories = mood_to_category.get(mood, ['Pizza', 'Pasta', 'Rice / Biryani'])
    
    # Get diverse items from different categories
    all_candidates = []
    
    for category in popular_categories:
        category_items = [item for item in menu_items if item['category'] == category]
        
        # Filter by diet preference
        if diet_type == 'veg':
            category_items = [item for item in category_items if not item['nc']]
        elif diet_type == 'non-veg':
            category_items = [item for item in category_items if item['nc']]
        
        all_candidates.extend(category_items)
    
    # Add variety by shuffling
    if all_candidates:
        random.shuffle(all_candidates)
        return all_candidates[0]
    
    # Ultimate fallback with variety
    if diet_type == 'veg':
        fallback = [item for item in menu_items if not item['nc']]
    elif diet_type == 'non-veg':
        fallback = [item for item in menu_items if item['nc']]
    else:
        fallback = menu_items
    
    if fallback:
        random.shuffle(fallback)
        return fallback[0]
    
    return None

def get_smart_recommendations_with_categories(mood, diet_type, categories, text=''):
    """Get recommendations using dynamic categories"""
    import time
    random.seed(int(time.time() * 1000))
    
    recommendations = []
    
    # Check for specific food requests first
    is_food_intent, detected_foods = detect_food_intent(text)
    
    if is_food_intent and detected_foods:
        for food_type in detected_foods:
            specific_items = get_specific_food_items(food_type, diet_type)
            recommendations.extend(specific_items[:2])
    
    # Use dynamic categories
    random.shuffle(categories)
    
    for category in categories:
        if len(recommendations) >= 10:
            break
            
        category_items = [item for item in menu_items if item['category'] == category]
        
        if diet_type == 'veg':
            category_items = [item for item in category_items if not item['nc']]
        elif diet_type == 'non-veg':
            category_items = [item for item in category_items if item['nc']]
        
        if category_items:
            random.shuffle(category_items)
            recommendations.extend(category_items[:1])
    
    # Remove duplicates
    seen = set()
    unique_recommendations = []
    for item in recommendations:
        if item['item_name'] not in seen:
            unique_recommendations.append(item)
            seen.add(item['item_name'])
    
    random.shuffle(unique_recommendations)
    return unique_recommendations[:10]

def get_smart_recommendations(mood, diet_type, text=''):
    """Get smart recommendations with variety and better selection"""
    import time
    random.seed(int(time.time() * 1000))  # Use current time for true randomness
    
    recommendations = []
    
    # Check for specific food requests first
    is_food_intent, detected_foods = detect_food_intent(text)
    
    if is_food_intent and detected_foods:
        # Handle specific food requests
        for food_type in detected_foods:
            specific_items = get_specific_food_items(food_type, diet_type)
            recommendations.extend(specific_items[:3])  # Max 3 per food type
    
    # Always add mood-based items for variety
    mood_categories = mood_to_category.get(mood, ['Pizza', 'Pasta', 'Rice / Biryani'])
    
    # Shuffle categories for variety
    random.shuffle(mood_categories)
    
    # Get diverse items from different categories
    for category in mood_categories:
        if len(recommendations) >= 10:
            break
            
        category_items = [item for item in menu_items if item['category'] == category]
        
        # Filter by diet
        if diet_type == 'veg':
            category_items = [item for item in category_items if not item['nc']]
        elif diet_type == 'non-veg':
            category_items = [item for item in category_items if item['nc']]
        
        # Add variety - shuffle and take different items each time
        if category_items:
            random.shuffle(category_items)
            recommendations.extend(category_items[:2])  # 2 items per category
    
    # Ensure we have good variety and remove duplicates
    seen_names = set()
    unique_recommendations = []
    
    for item in recommendations:
        if item['item_name'] not in seen_names:
            unique_recommendations.append(item)
            seen_names.add(item['item_name'])
    
    # If still not enough, add more popular items
    if len(unique_recommendations) < 8:
        all_categories = ['Rice / Biryani', 'Indian Main Course Non Veg', 'Indian Main Course Veg', 
                         'Pizza', 'Pasta', 'Chinese Non Veg', 'Chinese Veg', 'Smokey Tandoori Roasted Non Veg']
        
        random.shuffle(all_categories)
        
        for category in all_categories:
            if len(unique_recommendations) >= 10:
                break
                
            category_items = [item for item in menu_items 
                            if item['category'] == category and 
                            item['item_name'] not in seen_names]
            
            if diet_type == 'veg':
                category_items = [item for item in category_items if not item['nc']]
            elif diet_type == 'non-veg':
                category_items = [item for item in category_items if item['nc']]
            
            if category_items:
                random.shuffle(category_items)
                for item in category_items[:1]:  # Add 1 item per category
                    if len(unique_recommendations) < 10:
                        unique_recommendations.append(item)
                        seen_names.add(item['item_name'])
    
    # Final shuffle for complete randomness
    random.shuffle(unique_recommendations)
    return unique_recommendations[:10]  # Return max 10 recommendations

def get_specific_food_items(food_type, diet_type):
    """Get specific food items based on food type"""
    food_keywords = {
        'biryani': ['biryani', 'biriyani', 'pulao'],
        'pizza': ['pizza'],
        'pasta': ['pasta', 'penne'],
        'chicken': ['chicken', 'murgh'],
        'paneer': ['paneer'],
        'noodles': ['noodles', 'hakka', 'schezwan'],
        'momos': ['momos', 'momo'],
        'sandwich': ['sandwich'],
        'dosa': ['dosa'],
        'soup': ['soup'],
        'rice': ['rice', 'chawal'],
        'curry': ['curry', 'masala'],
        'tikka': ['tikka'],
        'kabab': ['kabab', 'kebab'],
        'fish': ['fish', 'seafood'],
        'egg': ['egg', 'omelette'],
        'mutton': ['mutton', 'lamb'],
        'prawn': ['prawn', 'shrimp'],
        'cold drinks': ['cold drink', 'pepsi', 'coke'],
        'juice': ['juice'],
        'coffee': ['coffee'],
        'tea': ['tea', 'chai'],
        'lassi': ['lassi'],
        'shake': ['shake', 'milkshake'],
        'smoothie': ['smoothie'],
        'thali': ['thali']
    }
    
    keywords = food_keywords.get(food_type, [food_type])
    matching_items = []
    
    for item in menu_items:
        item_name_lower = item['item_name'].lower()
        if any(keyword in item_name_lower for keyword in keywords):
            # Filter by diet
            if diet_type == 'veg' and item['nc']:
                continue
            elif diet_type == 'non-veg' and not item['nc']:
                continue
            matching_items.append(item)
    
    # Shuffle for variety
    random.shuffle(matching_items)
    return matching_items

def get_combo_items(main_item, diet_type):
    """Get combo items that go well with the main item - Main Course + Starter + Beverage structure"""
    if not main_item:
        return []
    
    main_name = main_item['item_name'].lower()
    main_category = main_item['category']
    combo_items = []
    
    # Define proper meal structure: Starter + Main Course + Beverage
    # Avoid bread with starters, no Hi Tea items
    
    # 1. Add a starter (appetizer) - never bread items
    starter_categories = ['Soup', 'Salad', 'Smokey Tandoori Starter Veg', 'Continental Starter', 'Bar Bite']
    if main_item['nc']:  # Non-veg main course
        starter_categories.extend(['Smokey Tandoori Roasted Non Veg'])
    
    # Shuffle starter categories for variety
    random.shuffle(starter_categories)
    
    for category in starter_categories:
        starter_items = [item for item in menu_items 
                        if item['category'] == category and 
                        item['item_name'] != main_item['item_name']]
        
        # Filter by diet but allow both for starters
        if diet_type == 'veg':
            starter_items = [item for item in starter_items if not item['nc']]
        elif diet_type == 'non-veg':
            starter_items = [item for item in starter_items if item['nc']]
        
        if starter_items:
            random.shuffle(starter_items)  # Add randomness
            combo_items.append(starter_items[0])
            break
    
    # 2. Add bread/accompaniment ONLY for main courses that need it
    if main_category in ['Indian Main Course Veg', 'Indian Main Course Non Veg']:
        bread_items = [item for item in menu_items 
                      if item['category'] == 'Indian Breads' and 
                      item['item_name'] != main_item['item_name']]
        if bread_items:
            random.shuffle(bread_items)
            combo_items.append(bread_items[0])
    
    # 3. Add beverage (always include)
    beverage_items = [item for item in menu_items 
                     if item['category'] == 'Beverages' and 
                     item['item_name'] != main_item['item_name']]
    
    # Filter beverages by diet if needed
    if diet_type == 'veg':
        beverage_items = [item for item in beverage_items if not item['nc']]
    elif diet_type == 'non-veg':
        beverage_items = [item for item in beverage_items if item['nc']]
    
    if beverage_items:
        random.shuffle(beverage_items)
        combo_items.append(beverage_items[0])
    
    # 4. Add dessert or side if we have space
    if len(combo_items) < 3:
        side_categories = ['Dessert', 'Raita']
        random.shuffle(side_categories)
        
        for category in side_categories:
            side_items = [item for item in menu_items 
                         if item['category'] == category and 
                         item['item_name'] != main_item['item_name']]
            
            if diet_type == 'veg':
                side_items = [item for item in side_items if not item['nc']]
            elif diet_type == 'non-veg':
                side_items = [item for item in side_items if item['nc']]
            
            if side_items:
                random.shuffle(side_items)
                combo_items.append(side_items[0])
                break
    
    return combo_items[:3]

@app.route('/test-mood', methods=['POST'])
def test_mood():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        if ML_AVAILABLE:
            ml_mood, ml_confidence = mood_detector.predict_mood_with_sentiment(text)
        else:
            ml_mood, ml_confidence = None, None
        
        simple_mood, simple_confidence = simple_mood_detect(text)
        
        return jsonify({
            "text": text,
            "ml_available": ML_AVAILABLE,
            "ml_result": {"mood": ml_mood, "confidence": ml_confidence} if ML_AVAILABLE else None,
            "simple_result": {"mood": simple_mood, "confidence": simple_confidence}
        })
    except Exception as e:
        logger.error(f"Test mood endpoint error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "menu_items": len(menu_items),
        "advanced_ml_available": ADVANCED_ML_AVAILABLE,
        "ml_available": ML_AVAILABLE
    })

if __name__ == '__main__':
    logger.info(f"Starting Mood-Based Food Recommendation API")
    logger.info(f"Menu items: {len(menu_items)}, Moods: {len(mood_to_category)}")
    logger.info(f"ML Model available: {ML_AVAILABLE}")
    
    # Production settings
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug)