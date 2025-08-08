import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import os

def load_model():
    """Load BERT model for prediction"""
    model_dir = 'models/baseline_bert'
    print(f"Loading model from {model_dir}")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model

def predict_sentence(tokenizer, model, sentence, device):
    """Predict the difficulty level of a sentence"""
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    
    return prediction

def parse_loop2_data():
    """Parse loop2_temp.txt and convert to loop_1_data.json format"""
    
    # Load BERT model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    tokenizer, model = load_model()
    model.to(device)
    model.eval()
    
    # Load loop2_temp_fixed.json
    with open('loop2_temp_fixed.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} items from loop2_temp_fixed.json")
    
    # Convert to loop_1_data.json format
    converted_data = []
    
    for item in tqdm(data, desc="Processing items"):
        original_id = item['sentence_id']
        original_sentence = item['original']
        original_prediction = int(item['original_prediction'])
        
        # Create counterfactuals list
        counterfactuals = []
        
        # Define the counterfactual types and their corresponding keys
        cf_types = {
            'lexical-frequency': 'lexical',
            'lexical-idiom': 'lexical',
            'restructure-passive': 'restructure',
            'restructure-participle': 'restructure',
            'resemantic-background': 'resemantic',
            'negation': 'negation',
            'insert-clause': 'insert',
            'delete-modifier': 'delete'
        }
        
        for cf_key, cf_type in cf_types.items():
            if cf_key in item:
                cf_sentence = item[cf_key]
                
                # Predict the difficulty level
                cf_prediction = predict_sentence(tokenizer, model, cf_sentence, device)
                
                # Determine if label is flipped
                is_label_flipped = (cf_prediction != original_prediction)
                
                counterfactual = {
                    'type': cf_type,
                    'sentence': cf_sentence,
                    'prediction': cf_prediction,
                    'is_label_flipped': is_label_flipped
                }
                counterfactuals.append(counterfactual)
        
        # Create the converted item
        converted_item = {
            'original_id': original_id,
            'original_sentence': original_sentence,
            'original_prediction': original_prediction,
            'counterfactuals': counterfactuals,
            'original_prediction_ground_truth': original_prediction
        }
        
        converted_data.append(converted_item)
    
    # Save the converted data
    output_path = 'data/Loop/loop_2/loop_2_data_full.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted data saved to {output_path}")
    print(f"Total items: {len(converted_data)}")
    
    # Print statistics
    total_counterfactuals = sum(len(item['counterfactuals']) for item in converted_data)
    label_flipped_count = sum(
        sum(1 for cf in item['counterfactuals'] if cf['is_label_flipped'])
        for item in converted_data
    )
    
    print(f"Total counterfactuals: {total_counterfactuals}")
    print(f"Label flipped counterfactuals: {label_flipped_count} ({label_flipped_count/total_counterfactuals*100:.1f}%)")
    
    return converted_data

if __name__ == "__main__":
    parse_loop2_data() 