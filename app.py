# Mood-based Food Recommendation API - Production Version
from flask import Flask, jsonify, request
import json
import re
import logging
import os
from datetime import datetime
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

# Try to import ML detector, fallback to simple detection
try:
    from mood_detector import ImprovedMoodDetector
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

app = Flask(__name__)

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
        'cold drinks': ['cold drink', 'cold drinks', 'soft drink', 'soda'],
        'juice': ['juice', 'fresh juice'],
        'coffee': ['coffee'],
        'tea': ['tea', 'chai'],
        'lassi': ['lassi'],
        'shake': ['shake', 'milkshake'],
        'smoothie': ['smoothie']
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
    mood_words = ['happy', 'sad', 'angry', 'stressed', 'relaxed', 'hungry', 'feeling', 'mood']
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
    hunger_phrases = ['feeling hungry', 'am hungry', 'getting hungry', 'want food', 'need food', 'craving food']
    if any(phrase in text_lower for phrase in hunger_phrases):
        return 'hungry', 0.8
    
    # Check for food mentions that indicate hunger
    food_mentions = ['biryani', 'pizza', 'pasta', 'noodles', 'chicken', 'rice', 'curry', 'sandwich']
    hunger_indicators = ['want', 'need', 'craving']
    
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

# Initialize mood detector
if ML_AVAILABLE:
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

# Mood to food category mapping
mood_to_category = {
    "happy": ["Beverages", "Dessert", "Ice Cream", "Hi Tea", "Pizza"],
    "sad": ["Soup", "Indian Main Course Veg", "Indian Main Course Non Veg", "Dessert", "Ice Cream"],
    "angry": ["Chinese Non Veg", "Smokey Tandoori Roasted Non Veg", "Bar Bite"],
    "stressed": ["Beverages", "Hi Tea", "Soup", "Dessert"],
    "relaxed": ["Salad", "South Indian", "Japanese", "Raita", "Beverages"],
    "hungry": ["Pizza", "Pasta", "Rice / Biryani", "Indian Main Course Veg", "Indian Main Course Non Veg", "Sandwich"],
    "adventurous": ["Japanese", "Continental Sizzler", "Fish / Prawns", "Continental Starter"],
    "energetic": ["Chinese Veg", "Chinese Non Veg", "Fried Rice / Noodles", "Steam Momos"],
    "comfort": ["Indian Main Course Veg", "Indian Main Course Non Veg", "Soup", "Indian Breads"],
    "light": ["Salad", "Soup", "South Indian", "Raita", "Hi Tea"],
    "spicy": ["Chinese Non Veg", "Smokey Tandoori Roasted Non Veg", "Smokey Tandoori Starter Veg"],
    "quick": ["Sandwich", "Wrap", "Kathi Roll", "Steam Momos", "Bar Bite"],
    "festive": ["Rice / Biryani", "Smokey Tandoori Roasted Non Veg", "Smokey Tandoori Starter Veg", "Dessert"],
    "healthy": ["Salad", "Soup", "South Indian", "Japanese", "Raita"],
    "indulgent": ["Pizza", "Pasta", "Continental Sizzler", "Dessert", "Ice Cream"],
    "exotic": ["Japanese", "Continental Sizzler", "Fish / Prawns", "Continental Starter"],
    "traditional": ["Indian Main Course Veg", "Indian Main Course Non Veg", "Indian Breads", "South Indian"],
    "party": ["Pizza", "Chinese Non Veg", "Continental Sizzler", "Beverages", "Dessert"],
    "romantic": ["Continental Sizzler", "Continental Starter", "Dessert", "Beverages"]
}

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

    categories = mood_to_category.get(mood, [])
    if not categories:
        return jsonify({"error": f"No category mapping for mood: {mood}"}), 404

    diet_type = data.get("diet_type", "both").lower()
    
    # Filter by categories
    recommendations = [item for item in menu_items if item["category"] in categories]
    
    # Enhanced food item detection and prioritization
    prioritized_items = []
    mentioned_foods = []
    
    if 'text' in data:
        text_lower = data['text'].lower()
        
        # Comprehensive food keyword mapping with exact menu item matching
        food_keywords = {
            # Rice/Biryani items
            'biryani': ['biryani', 'biriyani', 'pulao', 'pilaf'],
            'fried rice': ['fried rice', 'hakka rice', 'schezwan rice'],
            'rice': ['rice', 'chawal'],
            
            # Pizza & Italian
            'pizza': ['pizza'],
            'pasta': ['pasta', 'spaghetti', 'penne', 'macaroni'],
            
            # Chinese
            'noodles': ['noodles', 'hakka', 'schezwan', 'chowmein'],
            'momos': ['momos', 'momo', 'dumpling'],
            'manchurian': ['manchurian'],
            
            # Indian Breads & Snacks
            'sandwich': ['sandwich'],
            'dosa': ['dosa', 'uttapam'],
            'paratha': ['paratha', 'roti', 'naan', 'chapati'],
            'samosa': ['samosa'],
            'pakora': ['pakora', 'bhajiya'],
            
            # Soups & Starters
            'soup': ['soup'],
            'tikka': ['tikka'],
            'kabab': ['kabab', 'kebab', 'seekh'],
            'tandoori': ['tandoori'],
            
            # Curries & Main Course
            'curry': ['curry', 'masala', 'gravy'],
            'dal': ['dal', 'lentil'],
            'paneer': ['paneer', 'cottage cheese'],
            'butter chicken': ['butter chicken', 'murgh makhani'],
            
            # Proteins
            'chicken': ['chicken', 'murgh'],
            'fish': ['fish', 'seafood', 'machli'],
            'egg': ['egg', 'omelette', 'anda'],
            'mutton': ['mutton', 'lamb', 'goat'],
            'prawn': ['prawn', 'shrimp', 'jhinga'],
            
            # Beverages & Desserts
            'tea': ['tea', 'chai'],
            'coffee': ['coffee'],
            'lassi': ['lassi'],
            'ice cream': ['ice cream', 'kulfi'],
            'dessert': ['dessert', 'sweet', 'mithai'],
            
            # Salads & Light
            'salad': ['salad'],
            'raita': ['raita'],
            
            # Beverages & Drinks
            'cold drinks': ['cold drink', 'cold drinks', 'soft drink', 'soda', 'coke', 'pepsi'],
            'juice': ['juice', 'fresh juice'],
            'coffee': ['coffee', 'cappuccino', 'latte'],
            'tea': ['tea', 'chai', 'green tea'],
            'lassi': ['lassi', 'buttermilk'],
            'shake': ['shake', 'milkshake'],
            'smoothie': ['smoothie'],
            
            # Diet preferences
            'veg': ['veg', 'vegetarian'],
            'non-veg': ['non-veg', 'nonveg', 'non vegetarian', 'meat']
        }
        
        # Find all mentioned food items with enhanced matching
        for food_type, keywords in food_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                mentioned_foods.append(food_type)
                
                # Find matching items with multiple strategies
                matching_items = []
                
                # Strategy 1: Direct keyword match in item name
                for item in recommendations:
                    item_name_lower = item['item_name'].lower()
                    if any(keyword in item_name_lower for keyword in keywords):
                        matching_items.append(item)
                
                # Strategy 2: Category-based matching for specific foods
                category_mapping = {
                    'biryani': ['Rice / Biryani'],
                    'pizza': ['Pizza'],
                    'pasta': ['Pasta'],
                    'noodles': ['Fried Rice / Noodles', 'Chinese Veg', 'Chinese Non Veg'],
                    'momos': ['Steam Momos'],
                    'sandwich': ['Sandwich'],
                    'dosa': ['South Indian'],
                    'soup': ['Soup'],
                    'chicken': ['Indian Main Course Non Veg', 'Smokey Tandoori Roasted Non Veg', 'Chinese Non Veg'],
                    'paneer': ['Indian Main Course Veg'],
                    'fish': ['Fish / Prawns'],
                    'egg': ['Indian Main Course Non Veg'],
                    'mutton': ['Indian Main Course Non Veg', 'Smokey Tandoori Roasted Non Veg'],
                    'ice cream': ['Ice Cream'],
                    'dessert': ['Dessert'],
                    'salad': ['Salad'],
                    'cold drinks': ['Beverages'],
                    'juice': ['Beverages'],
                    'coffee': ['Beverages'],
                    'tea': ['Beverages'],
                    'lassi': ['Beverages'],
                    'shake': ['Beverages'],
                    'smoothie': ['Beverages'],
                    'rice': ['Rice / Biryani', 'Fried Rice / Noodles'],
                    'curry': ['Indian Main Course Veg', 'Indian Main Course Non Veg'],
                    'tikka': ['Smokey Tandoori Starter Veg', 'Smokey Tandoori Roasted Non Veg'],
                    'kabab': ['Smokey Tandoori Roasted Non Veg'],
                    'prawn': ['Fish / Prawns'],
                    'dal': ['Indian Main Course Veg']
                }
                
                if food_type in category_mapping:
                    for category in category_mapping[food_type]:
                        category_items = [item for item in menu_items if item['category'] == category]
                        matching_items.extend(category_items)
                
                # Special handling for noodles - also search all menu items
                if food_type == 'noodles':
                    noodle_items = [item for item in menu_items 
                                  if any(noodle_word in item['item_name'].lower() 
                                       for noodle_word in ['noodles', 'hakka', 'schezwan', 'chowmein'])]
                    matching_items.extend(noodle_items)
                
                # Remove duplicates and add to prioritized items
                unique_matches = []
                seen_names = set()
                for item in matching_items:
                    if item['item_name'] not in seen_names:
                        seen_names.add(item['item_name'])
                        unique_matches.append(item)
                
                prioritized_items.extend(unique_matches)
        
        # Enhanced reorganization for food-specific requests
        if prioritized_items:
            seen = set()
            final_priority = []
            
            # Add prioritized items first (food-specific matches)
            for item in prioritized_items:
                if item['item_name'] not in seen:
                    seen.add(item['item_name'])
                    final_priority.append(item)
            
            # Check if this is a specific food request (should only show that food)
            is_specific_food_request = any(word in text_lower for word in ['only', 'just', 'specifically']) or \
                                     any(food in text_lower for food in ['biryani', 'pizza', 'pasta', 'paneer', 'chicken', 'noodles', 'sandwich', 'dosa', 'momos', 'tikka', 'kabab', 'fish', 'egg', 'mutton', 'prawn', 'cold drink', 'juice', 'coffee', 'tea', 'lassi', 'shake', 'smoothie'])
            
            if is_specific_food_request and final_priority:
                # Only return food-specific matches for specific food requests
                recommendations = final_priority
            else:
                # Add remaining items from mood-based categories for general requests
                for item in recommendations:
                    if item['item_name'] not in seen:
                        final_priority.append(item)
                recommendations = final_priority
            
        # Additional category boost based on mentioned foods
        if mentioned_foods:
            category_boost = []
            if any(food in mentioned_foods for food in ['biryani', 'rice', 'fried rice']):
                category_boost.append('Rice / Biryani')
            if any(food in mentioned_foods for food in ['pizza']):
                category_boost.append('Pizza')
            if any(food in mentioned_foods for food in ['pasta']):
                category_boost.append('Pasta')
            if any(food in mentioned_foods for food in ['noodles', 'momos', 'manchurian']):
                category_boost.extend(['Fried Rice / Noodles', 'Steam Momos', 'Chinese Veg', 'Chinese Non Veg'])
            if any(food in mentioned_foods for food in ['chicken', 'mutton', 'fish']):
                category_boost.extend(['Indian Main Course Non Veg', 'Smokey Tandoori Roasted Non Veg'])
            
            # Add items from boosted categories only for non-specific requests
            is_specific_food_request = any(word in text_lower for word in ['only', 'just', 'specifically']) or \
                                     any(food in text_lower for food in ['biryani', 'pizza', 'pasta', 'paneer', 'chicken', 'noodles', 'sandwich', 'dosa', 'momos', 'tikka', 'kabab', 'fish', 'egg', 'mutton', 'prawn', 'cold drink', 'juice', 'coffee', 'tea', 'lassi', 'shake', 'smoothie'])
            if not is_specific_food_request:
                for category in category_boost:
                    category_items = [item for item in menu_items 
                                    if item['category'] == category and item['item_name'] not in [r['item_name'] for r in recommendations]]
                    recommendations.extend(category_items[:3])  # Add top 3 from each boosted category
    
    # Filter by diet preference (keep both veg and non-veg when "both")
    if diet_type == "veg":
        recommendations = [item for item in recommendations if not item["nc"]]
    elif diet_type == "non-veg":
        recommendations = [item for item in recommendations if item["nc"]]
    
    return jsonify({
        "detected_mood": mood,
        "confidence": round(confidence, 2),
        "categories": categories,
        "mentioned_foods": mentioned_foods if 'text' in data else [],
        "diet_type": diet_type,
        "recommendations": recommendations,
        "total_recommendations": len(recommendations)
    })

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
    
    # Get single main item
    main_item = get_single_main_item(text if 'text' in data else '', mood, diet_type)
    if not main_item:
        return jsonify({"error": "No suitable main item found"}), 404
    
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

def get_single_main_item(text, mood, diet_type):
    """Get single best main item based on text and mood"""
    text_lower = text.lower() if text else ''
    
    # Food-specific selection - find exact matches first
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
            for item in menu_items:
                item_name_lower = item['item_name'].lower()
                if any(keyword in item_name_lower for keyword in keywords):
                    if ((diet_type == 'veg' and not item['nc']) or 
                        (diet_type == 'non-veg' and item['nc']) or 
                        diet_type == 'both'):
                        matching_items.append(item)
            
            # Return the first matching item
            if matching_items:
                return matching_items[0]
    
    # If no specific food mentioned, use mood-based selection
    if not text_lower or not any(any(keyword in text_lower for keyword in keywords) for keywords in food_keywords.values()):
        mood_items = {
            'hungry': ['BUTTER CHICKEN', 'PANEER LABABDAR', 'EGG BIRYANI', 'PIZZA MARGARITA'],
            'happy': ['PIZZA MARGARITA', 'CHOCOLATE BROWNIE', 'PANEER TIKKA PIZZA'],
            'sad': ['BUTTER CHICKEN', 'DAL MAKHANI', 'CHOCOLATE BROWNIE'],
            'stressed': ['MASALA CHAI', 'CHOCOLATE BROWNIE', 'PANEER BUTTER MASALA'],
            'relaxed': ['MASALA CHAI', 'VEG PULAO', 'PANEER DOSA'],
            'comfort': ['DAL MAKHANI', 'BUTTER CHICKEN', 'PANEER BUTTER MASALA']
        }
        
        preferred_items = mood_items.get(mood, mood_items['hungry'])
        
        for item_name in preferred_items:
            item = next((item for item in menu_items if item_name.upper() in item['item_name'].upper()), None)
            if item and ((diet_type == 'veg' and not item['nc']) or 
                       (diet_type == 'non-veg' and item['nc']) or 
                       diet_type == 'both'):
                return item
    
    # Fallback
    fallback_items = [item for item in menu_items[:20] 
                     if ((diet_type == 'veg' and not item['nc']) or 
                        (diet_type == 'non-veg' and item['nc']) or 
                        diet_type == 'both')]
    return fallback_items[0] if fallback_items else None

def get_combo_items(main_item, diet_type):
    """Get combo items that go well with the main item"""
    if not main_item:
        return []
    
    main_category = main_item['category']
    main_name = main_item['item_name'].lower()
    is_main_veg = not main_item['nc']
    
    combo_rules = {
        # Main course combos
        'Indian Main Course Veg': ['Beverages', 'Indian Breads', 'Raita'],
        'Indian Main Course Non Veg': ['Beverages', 'Indian Breads', 'Raita'],
        'Rice / Biryani': ['Beverages', 'Raita', 'Dessert'],
        'Pizza': ['Beverages', 'Dessert'],
        'Pasta': ['Beverages', 'Salad'],
        'Sandwich': ['Beverages', 'Hi Tea'],
        'Chinese Veg': ['Beverages', 'Soup'],
        'Chinese Non Veg': ['Beverages', 'Soup'],
        'South Indian': ['Beverages', 'Hi Tea']
    }
    
    # Specific item combos
    specific_combos = {
        'biryani': ['MASALA CHAI', 'RAITA', 'GULAB JAMUN'],
        'pizza': ['COLD COFFEE', 'CHOCOLATE BROWNIE', 'COLD DRINK'],
        'pasta': ['COLD COFFEE', 'GARLIC BREAD', 'CAESAR SALAD'],
        'chicken': ['MASALA CHAI', 'NAAN', 'RAITA'],
        'paneer': ['MASALA CHAI', 'ROTI', 'RAITA'],
        'sandwich': ['MASALA CHAI', 'COLD COFFEE'],
        'burger': ['COLD DRINK', 'FRENCH FRIES'],
        'noodles': ['COLD DRINK', 'SOUP'],
        'dosa': ['FILTER COFFEE', 'COCONUT CHUTNEY']
    }
    
    combo_items = []
    
    # Add specific combos first
    for food_type, combo_names in specific_combos.items():
        if food_type in main_name:
            for combo_name in combo_names:
                combo_item = next((item for item in menu_items 
                                 if combo_name.upper() in item['item_name'].upper()), None)
                if combo_item and len(combo_items) < 3:
                    # Diet compatibility check
                    if ((diet_type == 'veg' and not combo_item['nc']) or 
                        (diet_type == 'non-veg' and combo_item['nc']) or 
                        diet_type == 'both'):
                        combo_items.append(combo_item)
    
    # Add category-based combos if we don't have enough specific combos
    if len(combo_items) < 3:
        combo_categories = combo_rules.get(main_category, ['Beverages'])
        
        for category in combo_categories:
            if len(combo_items) >= 3:
                break
            
            category_items = [item for item in menu_items 
                             if item['category'] == category and 
                             item['item_name'] != main_item['item_name'] and
                             item not in combo_items]
            
            # Filter by diet
            if diet_type == 'veg':
                category_items = [item for item in category_items if not item['nc']]
            elif diet_type == 'non-veg':
                category_items = [item for item in category_items if item['nc']]
            
            # Add first available item from category
            if category_items and len(combo_items) < 3:
                combo_items.append(category_items[0])
    
    return combo_items[:3]  # Return max 3 combo items

@app.route('/test-mood', methods=['POST'])
def test_mood():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "Text is required"}), 400
    
    try:
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
        logger.error(f"Recommend endpoint error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# Production error handlers
@app.errorhandler(404)
def not_found(error):
    logger.error(f"404 error: {request.url}")
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(400)
def bad_request(error):
    logger.error(f"400 error: {str(error)}")
    return jsonify({"error": "Bad request"}), 400

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "menu_items": len(menu_items),
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