# Mood-Based Food Recommendation System

AI-powered food recommendation system that analyzes user emotions from text and suggests appropriate food items.

## Features
- **Mood Detection**: ML model analyzes text to predict emotional states
- **Food Database**: 300+ items across 29 categories  
- **RESTful API**: Clean endpoints for mood detection and recommendations
- **19 Supported Moods**: happy, sad, angry, stressed, relaxed, hungry, adventurous, energetic, comfort, light, spicy, quick, festive, healthy, indulgent, exotic, traditional, party, romantic

## Quick Start

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Train model**
```bash
python improved_mood_detector.py
```

3. **Start API**
```bash
python app.py
```

4. **Test**
```bash
python test_api.py
```

## API Endpoints

- `GET /` - API status
- `POST /detect-mood` - Detect mood from text
- `POST /recommend` - Get food recommendations  
- `GET /categories` - List food categories
- `GET /moods` - List supported moods

## Usage Examples

**Mood Detection:**
```bash
curl -X POST http://127.0.0.1:5000/detect-mood \
  -H "Content-Type: application/json" \
  -d '{"text": "I'm feeling really happy today!"}'
```

**Food Recommendations:**
```bash
curl -X POST http://127.0.0.1:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{"text": "I'm stressed and need comfort food"}'
```

## Files
- `app.py` - Main Flask API
- `improved_mood_detector.py` - ML mood detection
- `ItemData.js` - Food database
- `test_api.py` - API testing
- `extract_categories.py` - Category analysis utility