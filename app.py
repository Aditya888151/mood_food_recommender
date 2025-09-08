# Mood-based Food Recommendation API
# Detects mood from text and recommends food items

from flask import Flask, jsonify, request
import json
import re
from textblob import TextBlob

# Try to import ML detector, fallback to simple detection
try:
    from improved_mood_detector import ImprovedMoodDetector
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

app = Flask(__name__)

# Initialize mood detector based on availability
if ML_AVAILABLE:
    mood_detector = ImprovedMoodDetector()
    try:
        mood_detector.load_model()
    except:
        print("Training mood detection model...")
        mood_detector.train_model()
else:
    # Simple keyword-based detection for Vercel
    mood_keywords = {
        'happy': ['happy', 'excited', 'great', 'wonderful', 'fantastic', 'amazing'],
        'sad': ['sad', 'down', 'depressed', 'blue', 'heartbroken', 'crying'],
        'angry': ['angry', 'mad', 'furious', 'rage', 'irritated', 'frustrated'],
        'stressed': ['stressed', 'overwhelmed', 'pressure', 'anxious', 'worried'],
        'relaxed': ['relaxed', 'calm', 'peaceful', 'zen', 'tranquil', 'chill'],
        'hungry': ['hungry', 'starving', 'famished', 'craving', 'need food'],
        'adventurous': ['adventurous', 'explore', 'new', 'different', 'unique'],
        'energetic': ['energetic', 'pumped', 'dynamic', 'charged', 'active'],
        'comfort': ['comfort', 'cozy', 'warm', 'familiar', 'homely'],
        'light': ['light', 'fresh', 'healthy', 'clean', 'simple']
    }
    
    def simple_mood_detect(text):
        text = text.lower()
        mood_scores = {}
        for mood, keywords in mood_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                mood_scores[mood] = score
        
        if mood_scores:
            best_mood = max(mood_scores, key=mood_scores.get)
            confidence = min(0.9, 0.5 + mood_scores[best_mood] * 0.1)
            return best_mood, confidence
        
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

def parse_js_data(file_path):
    """Parse JavaScript object notation from ItemData.js file"""
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract item properties using regex
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
    """
    Main recommendation endpoint - provides food recommendations based on mood.
    
    Accepts two types of input:
    1. Direct mood specification: {"mood": "happy"}
    2. Text for mood detection: {"text": "I'm feeling great today!"}
    
    Returns:
        JSON response with:
        - detected_mood: The identified or provided mood
        - confidence: Confidence score for mood detection (0-1)
        - categories: Food categories matching the mood
        - recommendations: List of food items
        - total_recommendations: Count of recommended items
    """
    data = request.get_json()
    
    # Check if mood is provided directly or needs to be detected from text
    if 'text' in data:
        # MOOD DETECTION FROM TEXT
        # Use machine learning model to analyze text and predict mood
        text = data.get('text', '')
        if not text:
            return jsonify({"error": "Text is required for mood detection"}), 400
        
        try:
            if ML_AVAILABLE:
                detected_mood, confidence = mood_detector.predict_mood_with_sentiment(text)
            else:
                detected_mood, confidence = simple_mood_detect(text)
            mood = detected_mood.lower()
        except Exception as e:
            return jsonify({"error": f"Mood detection failed: {str(e)}"}), 500
    else:
        # DIRECT MOOD SPECIFICATION
        # User provides mood directly without text analysis
        mood = data.get("mood", "").lower()
        confidence = 1.0  # Full confidence for direct mood input
        
    # Validate mood input
    if not mood:
        return jsonify({"error": "Mood is required (either directly or via text)"}), 400

    # MAP MOOD TO FOOD CATEGORIES
    # Look up which food categories match the detected/provided mood
    categories = mood_to_category.get(mood, [])
    if not categories:
        return jsonify({"error": f"No category mapping for mood: {mood}"}), 404

    # FILTER MENU ITEMS
    # Find all menu items that belong to the mood-appropriate categories
    recommendations = [item for item in menu_items if item["category"] in categories]
    
    # PREPARE RESPONSE
    # Structure the response with all relevant information
    response = {
        "detected_mood": mood,
        "confidence": round(confidence, 2),
        "categories": categories,
        "recommendations": recommendations,
        "total_recommendations": len(recommendations)
    }
    
    return jsonify(response)

@app.route('/categories', methods=['GET'])
def list_categories():
    """
    Get all available food categories from the menu.
    
    Returns:
        JSON array: Sorted list of all unique food categories
    """
    # Extract unique categories from all menu items
    categories = list(set(item["category"] for item in menu_items))
    return jsonify(sorted(categories))

@app.route('/moods', methods=['GET'])
def list_moods():
    """
    Get all available moods that the system can handle.
    
    Returns:
        JSON object with:
        - available_moods: List of all supported moods
        - total_moods: Count of supported moods
    """
    return jsonify({
        "available_moods": sorted(mood_to_category.keys()),
        "total_moods": len(mood_to_category)
    })

@app.route('/detect-mood', methods=['POST'])
def detect_mood():
    """
    Standalone mood detection endpoint.
    
    Input: {"text": "User's text describing their feelings"}
    
    Returns:
        JSON object with:
        - text: Original input text
        - detected_mood: Predicted mood
        - confidence: Confidence score (0-1)
        - available_categories: Food categories for this mood
    """
    data = request.get_json()
    text = data.get('text', '')
    
    # Validate input
    if not text:
        return jsonify({"error": "Text is required for mood detection"}), 400
    
    try:
        if ML_AVAILABLE:
            mood, confidence = mood_detector.predict_mood_with_sentiment(text)
        else:
            mood, confidence = simple_mood_detect(text)
        
        return jsonify({
            "text": text,
            "detected_mood": mood,
            "confidence": round(confidence, 2),
            "available_categories": mood_to_category.get(mood, [])
        })
    except Exception as e:
        return jsonify({"error": f"Mood detection failed: {str(e)}"}), 500

# For Vercel deployment
if __name__ == '__main__':
    print(f"Starting API with {len(menu_items)} items, {len(mood_to_category)} moods")
    app.run(debug=True)
else:
    # This is needed for Vercel
    application = app