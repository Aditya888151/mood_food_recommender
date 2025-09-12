# Mood-Based Food Recommendation API - User Manual

**Live API URL**: https://web-production-c87f9.up.railway.app

## Quick Start Guide

### 1. Test API Health
```bash
curl https://web-production-c87f9.up.railway.app/health
```
**Response**: `{"status": "healthy"}`

### 2. Get Food Recommendations
```bash
curl -X POST https://web-production-c87f9.up.railway.app/recommend \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling hungry", "diet_type": "non-veg"}'
```

## API Endpoints

### 1. Health Check
**URL**: `GET /health`
**Purpose**: Check if API is running
**Example**:
```bash
curl https://web-production-c87f9.up.railway.app/health
```

### 2. Get Recommendations
**URL**: `POST /recommend`
**Purpose**: Get multiple food recommendations based on mood
**Request Body**:
```json
{
  "text": "Your mood or food preference",
  "diet_type": "veg|non-veg|both"
}
```

**Examples**:

**Hungry Mood**:
```bash
curl -X POST https://web-production-c87f9.up.railway.app/recommend \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling hungry", "diet_type": "non-veg"}'
```

**Specific Food Request**:
```bash
curl -X POST https://web-production-c87f9.up.railway.app/recommend \
  -H "Content-Type: application/json" \
  -d '{"text": "I want biryani", "diet_type": "non-veg"}'
```

**Happy Mood**:
```bash
curl -X POST https://web-production-c87f9.up.railway.app/recommend \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling happy", "diet_type": "both"}'
```

### 3. Get Combo Meals
**URL**: `POST /combo`
**Purpose**: Get single main item + combo recommendations
**Request Body**: Same as `/recommend`

**Example**:
```bash
curl -X POST https://web-production-c87f9.up.railway.app/combo \
  -H "Content-Type: application/json" \
  -d '{"text": "I want chicken biryani", "diet_type": "non-veg"}'
```

### 4. List Categories
**URL**: `GET /categories`
**Purpose**: Get all available food categories
**Example**:
```bash
curl https://web-production-c87f9.up.railway.app/categories
```

### 5. List Moods
**URL**: `GET /moods`
**Purpose**: Get all supported moods
**Example**:
```bash
curl https://web-production-c87f9.up.railway.app/moods
```

## Supported Inputs

### Diet Types
- `"veg"` - Vegetarian only
- `"non-veg"` - Non-vegetarian only
- `"both"` - All options

### Supported Moods
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

### Text Input Examples
**Mood-based**:
- "I am feeling happy"
- "I am stressed"
- "Feeling sad today"

**Food-specific**:
- "I want biryani"
- "Give me pizza"
- "Need some chicken"
- "I want cold drinks"

**Combination**:
- "I'm hungry and want something spicy"
- "Feeling happy, want dessert"

## Response Format

### Recommend API Response
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

### Combo API Response
```json
{
  "detected_mood": "hungry",
  "confidence": 0.9,
  "main_item": {
    "item_name": "Chicken Biryani",
    "category": "Rice / Biryani",
    "qty": 1,
    "nc": true,
    "ttp": 250
  },
  "combo_items": [
    {
      "item_name": "Raita",
      "category": "Raita",
      "qty": 1,
      "nc": false,
      "ttp": 50
    }
  ],
  "diet_type": "non-veg"
}
```

## JavaScript/Web Integration

### Using Fetch API
```javascript
// Get recommendations
async function getRecommendations(text, dietType) {
  const response = await fetch('https://web-production-c87f9.up.railway.app/recommend', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text: text,
      diet_type: dietType
    })
  });
  
  const data = await response.json();
  return data;
}

// Example usage
getRecommendations("I am feeling hungry", "non-veg")
  .then(data => console.log(data));
```

### Using jQuery
```javascript
$.ajax({
  url: 'https://web-production-c87f9.up.railway.app/recommend',
  method: 'POST',
  contentType: 'application/json',
  data: JSON.stringify({
    text: "I want pizza",
    diet_type: "both"
  }),
  success: function(data) {
    console.log(data);
  }
});
```

## Python Integration

```python
import requests

# Get recommendations
def get_food_recommendations(text, diet_type):
    url = "https://web-production-c87f9.up.railway.app/recommend"
    payload = {
        "text": text,
        "diet_type": diet_type
    }
    response = requests.post(url, json=payload)
    return response.json()

# Example usage
result = get_food_recommendations("I am feeling hungry", "non-veg")
print(result)
```

## Testing with Postman

1. **Create New Request**
   - Method: POST
   - URL: `https://web-production-c87f9.up.railway.app/recommend`

2. **Set Headers**
   - Key: `Content-Type`
   - Value: `application/json`

3. **Set Body** (raw JSON)
   ```json
   {
     "text": "I am feeling hungry",
     "diet_type": "non-veg"
   }
   ```

4. **Send Request**

## Error Handling

### Common Errors
- **400 Bad Request**: Missing required fields
- **500 Internal Server Error**: Server processing error

### Error Response Format
```json
{
  "error": "Error description",
  "details": "Additional error details"
}
```

## Features

### Smart Food Detection
- Detects specific food requests: "I want biryani" → Returns only biryani items
- Mood-based suggestions: "I am happy" → Returns desserts, beverages
- Diet filtering: Automatically filters based on vegetarian/non-vegetarian preference

### Dynamic Categories
- Categories change with each request for variety
- Both `/recommend` and `/combo` APIs use dynamic category selection

### Multiple ML Models
- Advanced ML model with online food data training
- Enhanced model with comprehensive food pairing
- Simple keyword-based fallback system

## Menu Database
- **444+ food items** across **29 categories**
- Includes Indian, Chinese, Continental, Japanese cuisines
- Proper vegetarian/non-vegetarian classification
- Price information (ttp field)

## Rate Limits
- No rate limits currently implemented
- Recommended: Max 100 requests per minute per IP

## Support
For technical issues or questions, refer to the API logs or contact the development team.

## Version
**1.0.0** - Production Ready