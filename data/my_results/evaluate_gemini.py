import pandas as pd
import json
import re
import sys

# Define file path based on MODE argument (default to MODE 1)
mode = sys.argv[1] if len(sys.argv) > 1 else "1"
import os
# Adjust path based on current directory
if os.path.basename(os.getcwd()) == 'my_results':
    file_path = f'gemini_2.0_flash_kg_rag_based_mcq_{mode}_strategy.csv'
else:
    file_path = f'data/my_results/gemini_2.0_flash_kg_rag_based_mcq_{mode}_strategy.csv'

# Load the CSV file into DataFrame
df = pd.read_csv(file_path)

# Define an improved function to check if the correct answer is present in the LLM answer
def contains_correct_answer(row):
    try:
        llm_answer = str(row['llm_answer']).strip()
        correct_answer = str(row['correct_answer']).strip()

        # Direct match (most common case)
        if llm_answer == correct_answer:
            return True

        # Try to extract from JSON format
        # Pattern 1: {"answer": "VALUE"}
        json_match = re.search(r'\{[^{}]*"answer"\s*:\s*"([^"]+)"[^{}]*\}', llm_answer, re.DOTALL)
        if json_match:
            return json_match.group(1).strip() == correct_answer

        # Pattern 2: Clean JSON parsing after removing markdown
        try:
            cleaned = llm_answer
            if '```json' in cleaned:
                cleaned = cleaned.split('```json')[1]
            if '```' in cleaned:
                cleaned = cleaned.split('```')[0]
            cleaned = cleaned.strip()

            # Try parsing as JSON
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and 'answer' in parsed:
                return str(parsed['answer']).strip() == correct_answer
        except:
            pass

        # Fallback: Original method
        try:
            parsed = json.loads(llm_answer.replace('```', '').replace('\n', '').replace('json', '').replace('{{', '{').replace('}}', '}').split('}')[0] + '}')
            if 'answer' in parsed:
                return parsed['answer'] == correct_answer
        except:
            pass

        return False
    except Exception as e:
        return False

# Apply the function to each row
df['is_correct'] = df.apply(contains_correct_answer, axis=1)

# Calculate statistics
correct_count = df['is_correct'].sum()
total_count = len(df)
correct_rate = df['is_correct'].mean() * 100

print(f"\n=== Evaluation Results for MODE {mode} ===")
print(f"File: {file_path}")
print(f"Correct answers: {correct_count}/{total_count}")
print(f"Accuracy: {correct_rate:.2f}%")

# Check for API errors or failures
error_count = 0
api_error_examples = []

for idx, row in df.iterrows():
    llm_answer = str(row['llm_answer']).strip().lower()

    # Check for common API error patterns
    if any(error_indicator in llm_answer for error_indicator in ['error', 'rate limit', 'quota', 'timeout', 'failed', 'exception']):
        error_count += 1
        if len(api_error_examples) < 3:  # Keep first 3 examples
            api_error_examples.append({
                'question_idx': idx + 1,
                'question': row['question'][:50] + '...' if len(str(row['question'])) > 50 else row['question'],
                'answer': llm_answer[:100] + '...' if len(llm_answer) > 100 else llm_answer
            })

if error_count > 0:
    print(f"\n[WARNING] Found {error_count} potential API errors or failures")
    for example in api_error_examples:
        print(f"  Example {example['question_idx']}: {example['question']}")
        print(f"    Response: {example['answer']}")
else:
    print("\n[OK] No obvious API errors detected")

# Show sample of incorrect answers for debugging
print("\n=== Sample of Incorrect Answers (first 5) ===")
incorrect_df = df[~df['is_correct']].head(5)
for idx, row in incorrect_df.iterrows():
    print(f"Q{idx+1}:")
    print(f"  Correct: {row['correct_answer']}")
    print(f"  LLM: {row['llm_answer'][:100]}..." if len(str(row['llm_answer'])) > 100 else f"  LLM: {row['llm_answer']}")

