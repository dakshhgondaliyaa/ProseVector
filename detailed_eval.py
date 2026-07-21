import requests
import json
import time

TEST_QUERIES = [
    {
        "category": "Economics",
        "query": "An economics text that discusses the division of labor, productivity, and free markets, arguing that an 'invisible hand' guides self-interest to benefit society.",
        "expected": "The Wealth of Nations",
        "should_match": True
    },
    {
        "category": "Political Philosophy",
        "query": "A political text arguing that society requires a strong, absolute sovereign to avoid the 'war of all against all' and ensure order.",
        "expected": "Leviathan",
        "should_match": True
    },
    {
        "category": "Political Science",
        "query": "A French sociologist's observations on the American political system, equality, and civil society after visiting the United States in the 1830s.",
        "expected": "Democracy in America",
        "should_match": True
    },
    {
        "category": "Anthropology / History",
        "query": "An anthropological history book arguing that environmental and geographic factors, rather than intellectual or genetic superiority, allowed Eurasian societies to conquer others.",
        "expected": "Guns, Germs, and Steel",
        "should_match": True
    },
    {
        "category": "Philosophy of Science",
        "query": "A philosophy of science book that introduces the concept of a 'paradigm shift', arguing that science progresses through sudden revolutions rather than linear accumulation of facts.",
        "expected": "The Structure of Scientific Revolutions",
        "should_match": True
    },
    {
        "category": "Ethics",
        "query": "An ethical theory book arguing that the best action is the one that maximizes overall happiness or pleasure for the greatest number of people.",
        "expected": "Utilitarianism",
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
            res = requests.post(API_URL, json={"message": item["query"]}, timeout=120)
            data = res.json()
            answer = data.get("answer", "")
            
            # Simple check if the expected word/phrase is in the answer
            if item["should_match"]:
                if "wealth" in item["expected"].lower():
                    passed = "wealth of nations" in answer.lower()
                elif "leviathan" in item["expected"].lower():
                    passed = "leviathan" in answer.lower()
                elif "democracy" in item["expected"].lower():
                    passed = "democracy in america" in answer.lower()
                elif "guns" in item["expected"].lower():
                    passed = "guns, germs, and steel" in answer.lower() or "guns, germs" in answer.lower()
                elif "structure" in item["expected"].lower():
                    passed = "structure of scientific revolutions" in answer.lower() or "scientific revolutions" in answer.lower()
                elif "utilitarianism" in item["expected"].lower():
                    passed = "utilitarianism" in answer.lower()
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
