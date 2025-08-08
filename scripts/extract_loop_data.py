import json
import os

def extract_loop_data(input_file, output_file):
    """Extract original_id, original_sentence, and original_prediction from loop data"""
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load the original data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract only the required fields
    extracted_data = []
    for item in data:
        extracted_item = {
            'original_id': item['original_id'],
            'original_sentence': item['original_sentence'],
            'original_prediction': item['original_prediction']
        }
        extracted_data.append(extracted_item)
    
    # Save the extracted data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Extracted {len(extracted_data)} items from {input_file}")
    print(f"Saved to {output_file}")

def main():
    # Extract loop_2 data
    extract_loop_data(
        'outputs/analysis/loop_2_data.json',
        'data/Loop/loop_2/loop_2_data.json'
    )
    
    # Extract loop_3 data
    extract_loop_data(
        'outputs/analysis/loop_3_data.json',
        'data/Loop/loop_3/loop_3_data.json'
    )
    
    print("Extraction completed!")

if __name__ == "__main__":
    main() 