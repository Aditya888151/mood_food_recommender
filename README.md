# Mood-Based Food Recommendation API

AI-powered food recommendation system that suggests dishes based on user mood and dietary preferences.

## Features

- **Mood Detection**: Advanced ML models detect mood from text input
- **Dynamic Categories**: Different food categories every request for variety
- **Diet Filtering**: Vegetarian, Non-vegetarian, or Both options
- **Combo Recommendations**: Complete meal suggestions with sides
- **Multiple ML Models**: Enhanced, Advanced, and Simple fallback detection

## API Endpoints

### 1. Health Check
```
GET /health
```

### 2. Recommendations
```
POST /recommend
Content-Type: application/json

{
  "text": "I'm feeling hungry",
  "diet_type": "non-veg"
}
```

### 3. Combo Meals
```
POST /combo
Content-Type: application/json

{
  "text": "I want chicken biryani",
  "diet_type": "non-veg"
}
```

### 4. List Categories
```
GET /categories
```

### 5. List Moods
```
GET /moods
```

## Supported Moods

- **hungry** - Main courses, popular dishes
- **happy** - Desserts, beverages, celebratory food
- **sad** - Comfort food, warm dishes
- **angry** - Spicy food, bold flavors
- **stressed** - Calming beverages, light food
- **relaxed** - Healthy options, Japanese cuisine
- **energetic** - Chinese food, dynamic dishes
- **adventurous** - Exotic cuisines, new experiences
- **comfort** - Traditional dishes, familiar flavors
- **light** - Salads, healthy options

## Diet Types

- **veg** - Vegetarian only
- **non-veg** - Non-vegetarian only  
- **both** - All options

## Response Format

```json
{
  "detected_mood": "hungry",
  "confidence": 0.85,
  "categories": ["Rice / Biryani", "Pizza", "Chinese Non Veg"],
  "diet_type": "non-veg",
  "recommendations": [
    {
      "item_name": "Chicken Biryani",
      "category": "Rice / Biryani",
      "qty": 1,
      "nc": true,
      "ttp": 250
    }
  ],
  "total_recommendations": 10
}
```

## Deployment

### Local Development
```bash
pip install -r requirements.txt
python app.py
```

### Production Deployment

**Free Options:**
- **Railway**: `railway login && railway init && railway up`
- **Render**: Connect GitHub repo, set start command: `python app.py`
- **Fly.io**: `fly launch && fly deploy`

**Features:**
- Gunicorn WSGI server
- CORS enabled
- Environment-based configuration

## Menu Database

444+ food items across 29 categories including:
- Indian Main Courses (Veg/Non-Veg)
- Pizza & Pasta
- Chinese Cuisine
- Tandoori Items
- Beverages & Desserts
- And more...

## ML Models

1. **Enhanced Model** - Advanced Gradient Boosting with food pairing
2. **Advanced Model** - Comprehensive training with online data
3. **Simple Model** - Keyword-based fallback system

## API Testing

Test the API using tools like Postman or curl:

```bash
# Health check
curl https://your-app.com/health

# Get recommendations
curl -X POST https://your-app.com/recommend \
  -H "Content-Type: application/json" \
  -d '{"text": "I'm feeling hungry", "diet_type": "non-veg"}'
```

## Version

1.0.0 - Production Ready