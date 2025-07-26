import json
from datasets import load_dataset
from typing import List, Dict, Any

def convert_ag_news_to_sharegpt(split: str = "train", max_samples: int = None) -> List[Dict[str, Any]]:
    """
    Convert AG News dataset to ShareGPT format.
    
    Args:
        split: Dataset split to use ("train" or "test")
        max_samples: Maximum number of samples to convert (None for all)
    
    Returns:
        List of conversations in ShareGPT format
    """
    
    # Load the AG News dataset
    print(f"Loading AG News dataset ({split} split)...")
    dataset = load_dataset("wangrongsheng/ag_news", split=split)
    
    # Label mapping
    label_names = {
        0: "World",
        1: "Sports", 
        2: "Business",
        3: "Sci/Tech"
    }
    
    # Limit samples if specified
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    print(f"Converting {len(dataset)} samples to ShareGPT format...")
    
    sharegpt_conversations = []
    
    for i, example in enumerate(dataset):
        text = example["text"]
        label_id = example["label"]
        label_name = label_names[label_id]
        
        # Create the classification prompt
        prompt = f"Classify the following text into one of these categories: World, Sports, Business, or Sci/Tech.\n\nText: {text}"
        
        # Create the response
        response = f"The text is of type {label_name}."
        
        # Create ShareGPT conversation format
        conversation = {
            "conversations": [
                {
                    "from": "human",
                    "value": prompt
                },
                {
                    "from": "gpt", 
                    "value": response
                }
            ]
        }
        
        sharegpt_conversations.append(conversation)
        
        # Progress indicator
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1} samples...")
    
    return sharegpt_conversations

def save_sharegpt_format(conversations: List[Dict[str, Any]], filename: str):
    """Save conversations to JSON file in ShareGPT format."""
    print(f"Saving to {filename}...")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(conversations)} conversations to {filename}")

def main():
    """Main conversion pipeline."""
    
    # Convert training set
    print("Converting training set...")
    train_conversations = convert_ag_news_to_sharegpt(split="train")
    save_sharegpt_format(train_conversations, "ag_news_train_sharegpt.json")
    
    # Convert test set
    print("\nConverting test set...")
    test_conversations = convert_ag_news_to_sharegpt(split="test")
    save_sharegpt_format(test_conversations, "ag_news_test_sharegpt.json")
    
    # Show sample conversation
    print("\nSample conversation:")
    print(json.dumps(train_conversations[0], indent=2, ensure_ascii=False))
    
    print(f"\nConversion complete!")
    print(f"Training samples: {len(train_conversations)}")
    print(f"Test samples: {len(test_conversations)}")

if __name__ == "__main__":
    main()

# Alternative: Convert with custom parameters
def convert_custom(split="train", max_samples=1000, output_file="ag_news_custom_sharegpt.json"):
    """
    Convert with custom parameters for quick testing.
    
    Example usage:
    convert_custom(split="train", max_samples=1000, output_file="ag_news_sample.json")
    """
    conversations = convert_ag_news_to_sharegpt(split=split, max_samples=max_samples)
    save_sharegpt_format(conversations, output_file)
    return conversations

# Example of how to run with different prompt variations
def convert_with_custom_prompt(split: str = "train", max_samples: int = None) -> List[Dict[str, Any]]:
    """
    Convert with a more detailed classification prompt.
    """
    dataset = load_dataset("wangrongsheng/ag_news", split=split)
    
    label_names = {
        0: "World",
        1: "Sports", 
        2: "Business",
        3: "Sci/Tech"
    }
    
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    sharegpt_conversations = []
    
    for example in dataset:
        text = example["text"]
        label_id = example["label"]
        label_name = label_names[label_id]
        
        # More detailed prompt
        prompt = f"""Classify the following news article text into one of these four categories:

1. World - International news, politics, global events
2. Sports - Athletic events, games, sports news  
3. Business - Financial news, markets, corporate news
4. Sci/Tech - Science, technology, innovation news

Text to classify:
{text}

Please respond with just the category name."""
        
        response = f"The text is of type {label_name}."
        
        conversation = {
            "conversations": [
                {
                    "from": "human",
                    "value": prompt
                },
                {
                    "from": "gpt", 
                    "value": response
                }
            ]
        }
        
        sharegpt_conversations.append(conversation)
    
    return sharegpt_conversations