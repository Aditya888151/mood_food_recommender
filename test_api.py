import requests

base_url = "http://127.0.0.1:5000"

def test_mood_detection():
    """Test mood detection endpoint"""
    print("Testing Mood Detection...")
    
    test_texts = [
        "I'm feeling really happy and excited today!",
        "I'm so stressed about work",
        "I'm absolutely starving",
        "I want to try something new and adventurous"
    ]
    
    for text in test_texts:
        response = requests.post(f"{base_url}/detect-mood", json={"text": text})
        if response.status_code == 200:
            result = response.json()
            print(f"Text: '{text}' -> Mood: {result['detected_mood']} ({result['confidence']})")

def test_recommendations():
    """Test food recommendation endpoint"""
    print("\nTesting Recommendations...")
    
    # Test with text
    response = requests.post(f"{base_url}/recommend", json={"text": "I'm happy and want something sweet"})
    if response.status_code == 200:
        result = response.json()
        print(f"Mood: {result['detected_mood']}, Items: {result['total_recommendations']}")
    
    # Test with direct mood
    response = requests.post(f"{base_url}/recommend", json={"mood": "hungry"})
    if response.status_code == 200:
        result = response.json()
        print(f"Direct mood: hungry, Items: {result['total_recommendations']}")

def test_endpoints():
    """Test other endpoints"""
    print("\nTesting Other Endpoints...")
    
    # Categories
    response = requests.get(f"{base_url}/categories")
    if response.status_code == 200:
        categories = response.json()
        print(f"Categories: {len(categories)}")
    
    # Moods
    response = requests.get(f"{base_url}/moods")
    if response.status_code == 200:
        result = response.json()
        print(f"Moods: {result['total_moods']}")

if __name__ == "__main__":
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("API is running!")
            test_mood_detection()
            test_recommendations()
            test_endpoints()
        else:
            print("API not responding")
    except:
        print("Cannot connect to API. Start Flask app first.")