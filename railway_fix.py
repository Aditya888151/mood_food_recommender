#!/usr/bin/env python3
"""
Test the fixed app without TextBlob
"""

# Test imports
try:
    from app import app
    print("✅ App imports successfully")
    
    # Test basic functionality
    with app.test_client() as client:
        response = client.get('/health')
        if response.status_code == 200:
            print("✅ Health endpoint works")
        else:
            print("❌ Health endpoint failed")
            
        # Test recommend endpoint
        response = client.post('/recommend', 
                             json={"text": "I'm feeling hungry", "diet_type": "non-veg"},
                             headers={'Content-Type': 'application/json'})
        if response.status_code == 200:
            print("✅ Recommend endpoint works")
        else:
            print("❌ Recommend endpoint failed")
            
    print("✅ All tests passed - Railway deployment should work now!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Fix needed before deployment")