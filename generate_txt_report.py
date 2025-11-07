"""
Generate TXT format reports from CSV results
This script converts CSV results to comprehensive TXT reports with metrics and Q&A details
"""

import pandas as pd
import json
import re
import os
import sys
from datetime import datetime


def extract_answer_from_llm_response(llm_answer):
    """Extract the actual answer from LLM response (handles JSON format)"""
    try:
        llm_answer_str = str(llm_answer).strip()

        # Direct match
        if len(llm_answer_str) < 50 and llm_answer_str.replace('-', '').replace('_', '').replace(' ', '').isalnum():
            return llm_answer_str

        # Try to extract from JSON format
        json_match = re.search(r'\{[^{}]*"answer"\s*:\s*"([^"]+)"[^{}]*\}', llm_answer_str, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()

        # Try to parse as JSON
        try:
            cleaned = llm_answer_str
            if '```json' in cleaned:
                cleaned = cleaned.split('```json')[1]
            if '```' in cleaned:
                cleaned = cleaned.split('```')[0]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and 'answer' in parsed:
                return str(parsed['answer']).strip()
        except:
            pass

        # Return original if cannot parse
        return llm_answer_str[:100] + '...' if len(llm_answer_str) > 100 else llm_answer_str

    except Exception as e:
        return str(llm_answer)


def is_answer_correct(llm_answer, correct_answer):
    """Check if LLM answer matches correct answer"""
    try:
        llm_answer_str = str(llm_answer).strip()
        correct_answer_str = str(correct_answer).strip()

        # Direct match
        if llm_answer_str == correct_answer_str:
            return True

        # Try to extract from JSON
        json_match = re.search(r'\{[^{}]*"answer"\s*:\s*"([^"]+)"[^{}]*\}', llm_answer_str, re.DOTALL)
        if json_match:
            return json_match.group(1).strip() == correct_answer_str

        # Try JSON parsing
        try:
            cleaned = llm_answer_str
            if '```json' in cleaned:
                cleaned = cleaned.split('```json')[1]
            if '```' in cleaned:
                cleaned = cleaned.split('```')[0]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and 'answer' in parsed:
                return str(parsed['answer']).strip() == correct_answer_str
        except:
            pass

        return False
    except Exception as e:
        return False


def generate_txt_report(csv_file_path, output_txt_path, mode_name):
    """Generate comprehensive TXT report from CSV results"""

    # Read CSV file
    df = pd.read_csv(csv_file_path)

    # Calculate metrics
    df['is_correct'] = df.apply(lambda row: is_answer_correct(row['llm_answer'], row['correct_answer']), axis=1)
    df['extracted_answer'] = df['llm_answer'].apply(extract_answer_from_llm_response)

    total_count = len(df)
    correct_count = df['is_correct'].sum()
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0

    # Count errors
    error_count = df[df['llm_answer'].astype(str).str.lower().str.contains('error', na=False)].shape[0]

    # Generate report
    report = []
    report.append("=" * 80)
    report.append(f"KG-RAG EVALUATION REPORT - {mode_name}")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Source File: {os.path.basename(csv_file_path)}")
    report.append("")

    # Summary Statistics
    report.append("-" * 80)
    report.append("SUMMARY STATISTICS")
    report.append("-" * 80)
    report.append(f"Total Questions:        {total_count}")
    report.append(f"Correct Answers:        {correct_count}")
    report.append(f"Incorrect Answers:      {total_count - correct_count}")
    report.append(f"Accuracy:               {accuracy:.2f}%")
    report.append(f"Error Count:            {error_count}")
    report.append("")

    # Performance Assessment
    report.append("-" * 80)
    report.append("PERFORMANCE ASSESSMENT")
    report.append("-" * 80)
    if accuracy >= 75:
        assessment = "EXCELLENT - Exceeds baseline"
    elif accuracy >= 70:
        assessment = "GOOD - Comparable to baseline"
    elif accuracy >= 60:
        assessment = "FAIR - Below baseline, needs improvement"
    else:
        assessment = "POOR - Significant improvements needed"
    report.append(f"Assessment: {assessment}")
    report.append("")

    # Detailed Q&A Section
    report.append("=" * 80)
    report.append("DETAILED QUESTION & ANSWER ANALYSIS")
    report.append("=" * 80)
    report.append("")

    for idx, row in df.iterrows():
        question_num = idx + 1
        is_correct = row['is_correct']
        status = "[CORRECT]" if is_correct else "[INCORRECT]"

        report.append(f"{'=' * 80}")
        report.append(f"Question {question_num}/{total_count} - {status}")
        report.append(f"{'=' * 80}")
        report.append("")
        report.append(f"QUESTION:")
        report.append(f"  {row['question']}")
        report.append("")
        report.append(f"CORRECT ANSWER:")
        report.append(f"  {row['correct_answer']}")
        report.append("")
        report.append(f"LLM ANSWER:")
        report.append(f"  {row['extracted_answer']}")
        report.append("")

        if not is_correct:
            report.append(f"ANALYSIS:")
            if 'error' in str(row['llm_answer']).lower():
                report.append(f"  [WARNING] API Error detected")
            elif 'none' in str(row['extracted_answer']).lower():
                report.append(f"  [WARNING] LLM returned 'None' - possible context issue")
            else:
                report.append(f"  [WARNING] Incorrect answer selection")
            report.append("")

        report.append("")

    # Summary of incorrect answers
    incorrect_df = df[~df['is_correct']]
    if len(incorrect_df) > 0:
        report.append("=" * 80)
        report.append("SUMMARY OF INCORRECT ANSWERS")
        report.append("=" * 80)
        report.append("")

        for idx, row in incorrect_df.head(10).iterrows():
            report.append(f"Q{idx+1}: Expected '{row['correct_answer']}', Got '{row['extracted_answer'][:50]}...'")

        if len(incorrect_df) > 10:
            report.append(f"... and {len(incorrect_df) - 10} more incorrect answers")
        report.append("")

    # Write to file
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"[OK] Report generated: {output_txt_path}")
    print(f"   Total Questions: {total_count}")
    print(f"   Correct: {correct_count}")
    print(f"   Accuracy: {accuracy:.2f}%")

    return accuracy


def main():
    """Generate reports for all modes"""

    # Base paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'data', 'my_results')
    output_dir = os.path.join(base_dir, 'output')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Mode configurations
    modes = [
        ('0', 'MODE 0 - Baseline KG-RAG'),
        ('1', 'MODE 1 - JSON Structured'),
        ('2', 'MODE 2 - Prior Knowledge'),
        ('3', 'MODE 3 - Integrated')
    ]

    print("\n" + "=" * 80)
    print("GENERATING TXT REPORTS FOR ALL MODES")
    print("=" * 80 + "\n")

    results = []

    for mode_num, mode_name in modes:
        csv_file = os.path.join(results_dir, f'gemini_2.0_flash_kg_rag_based_mcq_{mode_num}_strategy.csv')
        txt_file = os.path.join(output_dir, f'mode_{mode_num}_report.txt')

        if os.path.exists(csv_file):
            print(f"\nProcessing {mode_name}...")
            accuracy = generate_txt_report(csv_file, txt_file, mode_name)
            results.append((mode_name, accuracy))
        else:
            print(f"\n[WARNING] {csv_file} not found, skipping...")

    # Generate comparison summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for mode_name, accuracy in results:
        print(f"{mode_name:40s} {accuracy:6.2f}%")

    print("\n[OK] All reports generated in: " + output_dir)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
