import json
import re

def fix_loop2_json():
    """Fix the JSON format in loop2_temp.txt"""
    
    # Read the file
    with open('loop2_temp.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove all closing brackets and opening brackets except the first and last
    # This will merge multiple JSON arrays into one
    content = re.sub(r'\]\s*\[', ',', content)
    
    # Write the fixed content
    with open('loop2_temp_fixed.json', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Try to parse the fixed JSON
    try:
        with open('loop2_temp_fixed.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully parsed {len(data)} items")
        return True
    except json.JSONDecodeError as e:
        print(f"Still has JSON error: {e}")
        return False

if __name__ == "__main__":
    fix_loop2_json() 