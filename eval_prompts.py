"""
Test prompts for evaluating recurrence experiments.
Each prompt has a question, expected answer, and category.
"""


PROMPTS = [
    # ── Math (10 problems) ─────────────────────────────────────────────
    {
        "id": "math_01",
        "category": "math",
        "question": "What is 23 * 47?",
        "answer": "1081",
        "prompt": "Question: What is 23 * 47?\nAnswer:",
    },
    {
        "id": "math_02",
        "category": "math",
        "question": "What is 156 + 289?",
        "answer": "445",
        "prompt": "Question: What is 156 + 289?\nAnswer:",
    },
    {
        "id": "math_03",
        "category": "math",
        "question": "What is 1000 - 387?",
        "answer": "613",
        "prompt": "Question: What is 1000 - 387?\nAnswer:",
    },
    {
        "id": "math_04",
        "category": "math",
        "question": "What is 84 / 7?",
        "answer": "12",
        "prompt": "Question: What is 84 / 7?\nAnswer:",
    },
    {
        "id": "math_05",
        "category": "math",
        "question": "What is 15 squared?",
        "answer": "225",
        "prompt": "Question: What is 15 squared?\nAnswer:",
    },
    {
        "id": "math_06",
        "category": "math",
        "question": "If a shirt costs $45 and is 20% off, what is the sale price?",
        "answer": "36",
        "prompt": "Question: If a shirt costs $45 and is 20% off, what is the sale price in dollars?\nAnswer:",
    },
    {
        "id": "math_07",
        "category": "math",
        "question": "What is the sum of the first 10 positive integers?",
        "answer": "55",
        "prompt": "Question: What is the sum of the first 10 positive integers?\nAnswer:",
    },
    {
        "id": "math_08",
        "category": "math",
        "question": "A rectangle has length 12 and width 5. What is its area?",
        "answer": "60",
        "prompt": "Question: A rectangle has length 12 and width 5. What is its area?\nAnswer:",
    },
    {
        "id": "math_09",
        "category": "math",
        "question": "What is 3^4?",
        "answer": "81",
        "prompt": "Question: What is 3 to the power of 4?\nAnswer:",
    },
    {
        "id": "math_10",
        "category": "math",
        "question": "If you have 3 apples and buy 4 bags of 6 apples each, how many apples total?",
        "answer": "27",
        "prompt": "Question: If you have 3 apples and buy 4 bags of 6 apples each, how many apples do you have in total?\nAnswer:",
    },
    # ── Logic (5 problems) ─────────────────────────────────────────────
    {
        "id": "logic_01",
        "category": "logic",
        "question": "All cats are animals. Some animals are pets. Can we conclude all cats are pets?",
        "answer": "no",
        "prompt": "Question: All cats are animals. Some animals are pets. Can we conclude that all cats are pets? Answer yes or no.\nAnswer:",
    },
    {
        "id": "logic_02",
        "category": "logic",
        "question": "If it rains, the ground is wet. The ground is wet. Did it rain?",
        "answer": "not necessarily",
        "prompt": "Question: If it rains, the ground gets wet. The ground is wet. Did it necessarily rain? Answer 'yes' or 'not necessarily'.\nAnswer:",
    },
    {
        "id": "logic_03",
        "category": "logic",
        "question": "What comes next: 2, 6, 18, 54, ?",
        "answer": "162",
        "prompt": "Question: What comes next in the sequence: 2, 6, 18, 54, ?\nAnswer:",
    },
    {
        "id": "logic_04",
        "category": "logic",
        "question": "A is taller than B. B is taller than C. Who is the shortest?",
        "answer": "C",
        "prompt": "Question: A is taller than B. B is taller than C. Who is the shortest?\nAnswer:",
    },
    {
        "id": "logic_05",
        "category": "logic",
        "question": "If all roses are flowers and some flowers fade quickly, can we say some roses fade quickly?",
        "answer": "no",
        "prompt": "Question: If all roses are flowers and some flowers fade quickly, can we necessarily conclude that some roses fade quickly? Answer yes or no.\nAnswer:",
    },
    # ── Factual recall (5 problems) ────────────────────────────────────
    {
        "id": "fact_01",
        "category": "factual",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "prompt": "Question: What is the capital of France?\nAnswer:",
    },
    {
        "id": "fact_02",
        "category": "factual",
        "question": "What planet is closest to the Sun?",
        "answer": "Mercury",
        "prompt": "Question: What planet is closest to the Sun?\nAnswer:",
    },
    {
        "id": "fact_03",
        "category": "factual",
        "question": "Who wrote Romeo and Juliet?",
        "answer": "Shakespeare",
        "prompt": "Question: Who wrote Romeo and Juliet?\nAnswer:",
    },
    {
        "id": "fact_04",
        "category": "factual",
        "question": "What is the chemical symbol for water?",
        "answer": "H2O",
        "prompt": "Question: What is the chemical symbol for water?\nAnswer:",
    },
    {
        "id": "fact_05",
        "category": "factual",
        "question": "How many continents are there?",
        "answer": "7",
        "prompt": "Question: How many continents are there on Earth?\nAnswer:",
    },
]


def get_prompts(category=None):
    """Get prompts, optionally filtered by category."""
    if category is None:
        return PROMPTS
    return [p for p in PROMPTS if p["category"] == category]


def check_answer(predicted: str, expected: str) -> bool:
    """
    Check if the predicted answer contains the expected answer.
    Case-insensitive, checks if expected appears anywhere in predicted.
    """
    predicted_lower = predicted.lower().strip()
    expected_lower = expected.lower().strip()
    return expected_lower in predicted_lower


if __name__ == "__main__":
    print(f"Total prompts: {len(PROMPTS)}")
    for cat in ["math", "logic", "factual"]:
        print(f"  {cat}: {len(get_prompts(cat))}")
    print("\nSample prompt:")
    print(PROMPTS[0]["prompt"])
