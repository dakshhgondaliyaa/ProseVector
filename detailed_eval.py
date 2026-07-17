import requests
import json
import time

TEST_QUERIES = [
    {
        "category": "Fantasy",
        "query": "An orphan boy discovers he is a wizard and attends a magical school, uncovering a plot to steal a stone that grants immortality.",
        "expected": "Harry Potter and the Philosopher's Stone",
        "should_match": True
    },
    {
        "category": "Sci-Fi",
        "query": "A young boy named Ender is sent to a military academy in space to prepare for a future alien invasion by the Formics.",
        "expected": "Ender's Game",
        "should_match": True
    },
    {
        "category": "Classic Literature",
        "query": "A wealthy man named Jay Gatsby throws lavish parties in Long Island to win back his former love, Daisy, in the 1920s.",
        "expected": "The Great Gatsby",
        "should_match": True
    },
    {
        "category": "Academic / Political",
        "query": "A political treatise that advises rulers on how to gain and maintain power. It begins by classifying different types of states.",
        "expected": "The Prince",
        "should_match": True
    },
    {
        "category": "Dystopian",
        "query": "A man works at the Ministry of Truth where he rewrites history for Big Brother in a totalitarian society.",
        "expected": "1984",
        "should_match": True
    },
    {
        "category": "Vague Summary",
        "query": "A hobbit inherits a powerful ring and has to travel to a volcano in Mordor to destroy it.",
        "expected": "The Fellowship of the Ring",
        "should_match": True
    },
    {
        "category": "Impossible Match",
        "query": "A book about an AI named Antigravity that helps users write python code and debug servers.",
        "expected": "I couldn't find an exact match",
        "should_match": False
    }
]

API_URL = "http://localhost:7860/chat"

def run_tests():
    results = []
    for item in TEST_QUERIES:
        print(f"Testing: {item['category']}")
        try:
            res = requests.post(API_URL, json={"message": item["query"]}, timeout=30)
            data = res.json()
            answer = data.get("answer", "")
            
            # Simple check if the expected word/phrase is in the answer
            if item["should_match"]:
                if "gatsby" in item["expected"].lower():
                    passed = "gatsby" in answer.lower()
                elif "ender" in item["expected"].lower():
                    passed = "ender" in answer.lower()
                elif "harry" in item["expected"].lower():
                    passed = "harry" in answer.lower()
                elif "prince" in item["expected"].lower():
                    passed = "prince" in answer.lower()
                elif "1984" in item["expected"].lower():
                    passed = "1984" in answer.lower()
                elif "fellowship" in item["expected"].lower():
                    passed = "fellowship" in answer.lower() or "lord of the rings" in answer.lower()
                else:
                    passed = item["expected"].lower() in answer.lower()
            else:
                passed = "couldn't find" in answer.lower() or "could not find" in answer.lower()

            results.append({
                "category": item["category"],
                "query": item["query"],
                "expected": item["expected"] if item["should_match"] else "Fallback Response",
                "actual_answer": answer,
                "passed": passed
            })
        except Exception as e:
            results.append({
                "category": item["category"],
                "query": item["query"],
                "expected": item["expected"],
                "actual_answer": f"ERROR: {str(e)}",
                "passed": False
            })
        time.sleep(1) # Be nice to local LLM

    # Write results to a JSON file for the agent to read
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Done! Results saved to eval_results.json")

if __name__ == "__main__":
    run_tests()
