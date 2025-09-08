import re
from collections import Counter

def extract_categories():
    """Extract categories from ItemData.js"""
    with open('ItemData.js', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all category values
    category_pattern = r'category:\s*["\']([^"\']+)["\']'
    categories = re.findall(category_pattern, content)
    
    return categories

if __name__ == "__main__":
    categories = extract_categories()
    unique_categories = sorted(set(categories))
    category_counts = Counter(categories)
    
    print(f"Total Items: {len(categories)}")
    print(f"Unique Categories: {len(unique_categories)}")
    print("\nAll Categories:")
    for i, category in enumerate(unique_categories, 1):
        print(f"{i:2d}. {category}")
    
    print("\nItems per Category:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{category}: {count} items")