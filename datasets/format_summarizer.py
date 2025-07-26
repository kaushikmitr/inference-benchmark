import json
from datasets import load_dataset
from typing import List, Dict, Any
import re

def convert_mediasum_to_sharegpt(split: str = "train", max_samples: int = None, config: str = "roberta_prepended") -> List[Dict[str, Any]]:
    """
    Convert MediaSum dataset to ShareGPT format for dialogue summarization.
    
    Args:
        split: Dataset split to use ("train", "validation", or "test")
        max_samples: Maximum number of samples to convert (None for all)
        config: Configuration to use ("roberta_prepended", "roberta", "newline", "bert", or "list")
    
    Returns:
        List of conversations in ShareGPT format
    """
    
    # Load the MediaSum dataset
    print(f"Loading MediaSum dataset ({split} split, config: {config})...")
    try:
        dataset = load_dataset("ccdv/mediasum", config, split=split, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative approach...")
        dataset = load_dataset("ccdv/mediasum", split=split, trust_remote_code=True)
    
    # Limit samples if specified
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    print(f"Converting {len(dataset)} samples to ShareGPT format...")
    
    sharegpt_conversations = []
    
    for i, example in enumerate(dataset):
        document = example["document"]
        summary = example["summary"]
        
        # Clean up the document text
        document = clean_dialogue_text(document)
        
        # Create the summarization prompt
        prompt = f"""Please provide a concise summary of the following dialogue/interview transcript:

{document}

Please summarize the key points, main topics discussed, and important information from this conversation."""
        
        # Create the response
        response = summary.strip()
        
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
        
        # Add metadata if available
        if "id" in example:
            conversation["id"] = example["id"]
        
        sharegpt_conversations.append(conversation)
        
        # Progress indicator
        if (i + 1) % 5000 == 0:
            print(f"Processed {i + 1} samples...")
    
    return sharegpt_conversations

def clean_dialogue_text(text: str) -> str:
    """
    Clean dialogue text for better readability.
    """
    if not text:
        return ""
    
    # Remove excessive whitespace and normalize line breaks
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Clean up common transcript artifacts
    text = text.replace('</s>', '\n')
    text = text.replace('[SEP]', '\n')
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def convert_with_instructional_prompt(split: str = "train", max_samples: int = None) -> List[Dict[str, Any]]:
    """
    Convert with more detailed instructional prompt for summarization.
    """
    print(f"Loading MediaSum dataset ({split} split) with instructional prompts...")
    try:
        dataset = load_dataset("ccdv/mediasum", "roberta_prepended", split=split, trust_remote_code=True)
    except:
        dataset = load_dataset("ccdv/mediasum", split=split, trust_remote_code=True)
    
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    sharegpt_conversations = []
    
    for i, example in enumerate(dataset):
        document = example["document"]
        summary = example["summary"]
        
        document = clean_dialogue_text(document)
        
        # More detailed instructional prompt
        prompt = f"""You are a professional summarization assistant. Please read the following media interview transcript and provide a clear, concise summary that captures:

1. The main topic or subject being discussed
2. Key points made by participants
3. Important facts, findings, or conclusions
4. Any significant quotes or insights

Here is the transcript to summarize:

{document}

Please provide a summary that would help someone quickly understand what this interview was about without needing to read the full transcript."""
        
        response = summary.strip()
        
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
        
        if "id" in example:
            conversation["id"] = example["id"]
        
        sharegpt_conversations.append(conversation)
        
        if (i + 1) % 5000 == 0:
            print(f"Processed {i + 1} samples...")
    
    return sharegpt_conversations

def convert_with_role_specific_prompt(split: str = "train", max_samples: int = None) -> List[Dict[str, Any]]:
    """
    Convert with role-specific prompts that emphasize different aspects of summarization.
    """
    print(f"Loading MediaSum dataset ({split} split) with role-specific prompts...")
    try:
        dataset = load_dataset("ccdv/mediasum", "roberta_prepended", split=split, trust_remote_code=True)
    except:
        dataset = load_dataset("ccdv/mediasum", split=split, trust_remote_code=True)
    
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
    
    # Different prompt styles for variety
    prompt_styles = [
        "As a journalist, summarize this interview highlighting the most newsworthy points:",
        "As a research assistant, provide a comprehensive summary of this media interview:",
        "As an editor, create a brief overview of this interview for publication:",
        "Summarize this interview transcript, focusing on the key information and main themes:",
        "Please analyze and summarize the following interview, highlighting important insights:"
    ]
    
    sharegpt_conversations = []
    
    for i, example in enumerate(dataset):
        document = example["document"]
        summary = example["summary"]
        
        document = clean_dialogue_text(document)
        
        # Rotate through different prompt styles
        prompt_prefix = prompt_styles[i % len(prompt_styles)]
        
        prompt = f"""{prompt_prefix}

{document}

Summary:"""
        
        response = summary.strip()
        
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
        
        if "id" in example:
            conversation["id"] = example["id"]
        
        sharegpt_conversations.append(conversation)
        
        if (i + 1) % 5000 == 0:
            print(f"Processed {i + 1} samples...")
    
    return sharegpt_conversations

def save_sharegpt_format(conversations: List[Dict[str, Any]], filename: str):
    """Save conversations to JSON file in ShareGPT format."""
    print(f"Saving to {filename}...")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(conversations)} conversations to {filename}")

def main():
    """Main conversion pipeline with multiple prompt variations."""
    
    # Convert training set with basic prompt
    print("Converting training set with basic summarization prompt...")
    train_conversations = convert_mediasum_to_sharegpt(split="train")
    save_sharegpt_format(train_conversations, "mediasum_train_basic_sharegpt.json")
    
    # Convert validation set
    print("\nConverting validation set...")
    val_conversations = convert_mediasum_to_sharegpt(split="validation")
    save_sharegpt_format(val_conversations, "mediasum_validation_sharegpt.json")
    
    # Convert test set
    print("\nConverting test set...")
    test_conversations = convert_mediasum_to_sharegpt(split="test")
    save_sharegpt_format(test_conversations, "mediasum_test_sharegpt.json")
    
    # Show sample conversation
    print("\nSample conversation (basic prompt):")
    print(json.dumps(train_conversations[0], indent=2, ensure_ascii=False))
    
    print(f"\nBasic conversion complete!")
    print(f"Training samples: {len(train_conversations)}")
    print(f"Validation samples: {len(val_conversations)}")
    print(f"Test samples: {len(test_conversations)}")

def create_instructional_dataset(max_samples_per_split: int = 1000):
    """Create a smaller dataset with instructional prompts for testing."""
    print("Creating instructional prompt dataset...")
    
    # Training set with instructional prompts
    train_instruct = convert_with_instructional_prompt(split="train", max_samples=max_samples_per_split)
    save_sharegpt_format(train_instruct, "mediasum_train_instructional_sharegpt.json")
    
    # Validation set
    val_instruct = convert_with_instructional_prompt(split="validation", max_samples=500)
    save_sharegpt_format(val_instruct, "mediasum_val_instructional_sharegpt.json")
    
    print("\nSample instructional conversation:")
    print(json.dumps(train_instruct[0], indent=2, ensure_ascii=False))

def create_role_specific_dataset(max_samples_per_split: int = 1000):
    """Create a dataset with role-specific prompts for variety."""
    print("Creating role-specific prompt dataset...")
    
    # Training set with role-specific prompts
    train_role = convert_with_role_specific_prompt(split="train", max_samples=max_samples_per_split)
    save_sharegpt_format(train_role, "mediasum_train_role_specific_sharegpt.json")
    
    print("\nSample role-specific conversation:")
    print(json.dumps(train_role[0], indent=2, ensure_ascii=False))

# Convenience function for quick testing
def convert_sample(split="validation", max_samples=100, prompt_type="basic"):
    """
    Convert a small sample for quick testing.
    
    Args:
        split: Dataset split ("train", "validation", "test")
        max_samples: Number of samples to convert
        prompt_type: Type of prompt ("basic", "instructional", "role_specific")
    """
    print(f"Converting {max_samples} samples from {split} split with {prompt_type} prompts...")
    
    if prompt_type == "instructional":
        conversations = convert_with_instructional_prompt(split=split, max_samples=max_samples)
    elif prompt_type == "role_specific":
        conversations = convert_with_role_specific_prompt(split=split, max_samples=max_samples)
    else:
        conversations = convert_mediasum_to_sharegpt(split=split, max_samples=max_samples)
    
    filename = f"mediasum_{split}_{prompt_type}_sample_{max_samples}.json"
    save_sharegpt_format(conversations, filename)
    
    print(f"\nSample conversation:")
    print(json.dumps(conversations[0], indent=2, ensure_ascii=False))
    
    return conversations

if __name__ == "__main__":
    # Run basic conversion
    main()
    
    # Optional: Create additional prompt variations
    # create_instructional_dataset(max_samples_per_split=5000)
    # create_role_specific_dataset(max_samples_per_split=5000)