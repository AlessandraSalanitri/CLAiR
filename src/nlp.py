from transformers import pipeline

# load NLP model 
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# define possible commands - momentarly hard coded
COMMAND_LABELS = [
    "check weather",
    "tell time",
    "tell a joke",
    "open YouTube",
    "play music",
    "search Google"
]

def understand_command(text):
    # process transcribed txt and identify user intent
    
    result = classifier(text, COMMAND_LABELS)
    
    # get the highest confidence label
    intent = result["labels"][0]
    confidence = result["scores"][0]
    
    print(f" Detected intent: {intent} (Confidence:{confidence:.2f})")
    
    return intent

# test nlp
if __name__ == "__main__":
    test_text = "Hey Clair, tell me a joke!"
    understand_command(test_text)