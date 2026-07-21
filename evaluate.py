import requests
import json

# 1. Define our Evaluation Dataset (Ground Truth)
# These are plots mapped to the exact book titles we expect the model to return.
EVAL_DATA = [
    {
        "query": "A man named Winston Smith works at the Ministry of Truth where he rewrites historical records. He lives in a totalitarian society ruled by Big Brother.",
        "expected_book": "1984"
    },
    {
        "query": "An eleven-year-old orphan discovers he is actually a wizard and is invited to attend a magical school. He uncovers a plot to steal a magical stone that grants immortality.",
        "expected_book": "Harry Potter and the Philosopher's Stone"
    },
    {
        "query": "A group of farm animals overthrow their human farmer, hoping to create a society where they can be equal, free, and happy.",
        "expected_book": "Animal Farm"
    },
    {
        "query": "A young girl named Alice falls through a rabbit hole into a fantasy world populated by peculiar, anthropomorphic creatures.",
        "expected_book": "Alice's Adventures in Wonderland"
    }
]

API_URL = "http://localhost:7860/chat"

def run_evaluation():
    print(f"Starting Evaluation on {len(EVAL_DATA)} queries...\n")
    
    correct_retrievals = 0
    correct_answers = 0

    for i, item in enumerate(EVAL_DATA):
        query = item["query"]
        expected = item["expected_book"]
        
        print(f"Test {i+1}:")
        print(f"Query: {query}")
        print(f"Expected Book: {expected}")
        
        try:
            # Send request to our Flask app
            response = requests.post(
                API_URL, 
                json={"message": query}, 
                headers={"Content-Type": "application/json"}
            )
            data = response.json()
            
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            
            # 1. Check Retrieval Accuracy
            # Did the embedding model find the book in the top 5 chunks?
            sources_text = " ".join(sources)
            retrieval_hit = expected.lower() in sources_text.lower()
            if retrieval_hit:
                correct_retrievals += 1
                
            # 2. Check Generation Accuracy
            # Did the LLM output the correct book?
            answer_hit = expected.lower() in answer.lower()
            if answer_hit:
                correct_answers += 1
                
            print(f"Retrieval Match: {'PASS' if retrieval_hit else 'FAIL'}")
            print(f"LLM Answer Match: {'PASS' if answer_hit else 'FAIL'}")
            print(f"LLM Answer: {answer}\n")
            
        except Exception as e:
            print(f"Error querying API: {e}\n")

    # Calculate final scores
    retrieval_accuracy = (correct_retrievals / len(EVAL_DATA)) * 100
    llm_accuracy = (correct_answers / len(EVAL_DATA)) * 100
    
    print("="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Retrieval Accuracy (Vector Search): {retrieval_accuracy}%")
    print(f"LLM Generation Accuracy: {llm_accuracy}%")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()
