from datasets import load_dataset
dataset = load_dataset("textminr/cmu-book-summaries", split="train")

print("Looking for Hobbes' Leviathan...")
found = False
for item in dataset:
    if "Leviathan" in item.get("title", ""):
        print(f"Found: {item.get('title')} by {item.get('author')}")
        if item.get("author") and "Hobbes" in item.get("author"):
            found = True
    
    if item.get("author") and "Hobbes" in item.get("author"):
        print(f"Found book by Hobbes: {item.get('title')}")

if not found:
    print("Hobbes' Leviathan is NOT in this dataset.")
