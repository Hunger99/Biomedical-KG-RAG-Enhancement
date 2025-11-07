'''
This script takes the MCQ style questions from the csv file and save the result as another csv file. 
Before running this script, make sure to configure the filepaths in config.yaml file.
Command line argument should be either 'gpt-4' or 'gpt-35-turbo'
'''

from kg_rag.utility import *
import sys
import time
import os
import shutil
from datetime import datetime


from tqdm import tqdm
CHAT_MODEL_ID = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "0"
START_INDEX = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # Allow starting from specific index

QUESTION_PATH = config_data["MCQ_PATH"]
SYSTEM_PROMPT = system_prompts["MCQ_QUESTION"]
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH"]
NODE_CONTEXT_PATH = config_data["NODE_CONTEXT_PATH"]
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]
TEMPERATURE = config_data["LLM_TEMPERATURE"]
SAVE_PATH = config_data["SAVE_RESULTS_PATH"]


CHAT_DEPLOYMENT_ID = CHAT_MODEL_ID

save_name = "_".join(CHAT_MODEL_ID.split("-"))+"_kg_rag_based_mcq_{mode}_strategy.csv"


vectorstore = load_chroma(VECTOR_DB_PATH, SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL)
embedding_function_for_context_retrieval = load_sentence_transformer(SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL)
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)
edge_evidence = False


### MODE 0: Original KG_RAG                     ### 
### MODE 1: jsonlize the context from KG search ### 
### MODE 2: Add the prior domain knowledge      ### 
### MODE 3: Combine MODE 1 & 2                  ### 

def create_backup_before_resume(mode, start_index):
    """Create timestamped backup before resuming from index"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"gemini_2.0_flash_kg_rag_based_mcq_{mode}_strategy.csv"
    output_path = os.path.join(SAVE_PATH, output_file)

    if os.path.exists(output_path):
        backup_dir = os.path.join(SAVE_PATH, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        backup_name = f"backup_mode{mode}_index{start_index}_{timestamp}.csv"
        backup_path = os.path.join(backup_dir, backup_name)
        shutil.copy2(output_path, backup_path)
        print(f"Backup created: {backup_path}")
        return backup_path
    return None

def save_checkpoint(answer_list, mode, question_num):
    """Save intermediate checkpoint"""
    checkpoint_df = pd.DataFrame(answer_list,
                                columns=["question", "correct_answer", "llm_answer"])
    checkpoint_dir = os.path.join(SAVE_PATH, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = f"checkpoint_mode{mode}_{question_num}.csv"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    checkpoint_df.to_csv(checkpoint_path, index=False)
    print(f"Checkpoint saved at question {question_num}")

def main():
    start_time = time.time()
    question_df = pd.read_csv(QUESTION_PATH)

    # Create backup if resuming
    if START_INDEX > 0:
        create_backup_before_resume(MODE, START_INDEX)
        print(f"Starting from index {START_INDEX} (skipping first {START_INDEX} questions)")
        question_df = question_df.iloc[START_INDEX:]

    answer_list = []
    CHECKPOINT_INTERVAL = 50  # Save checkpoint every 50 questions
    question_count = 0

    for index, row in tqdm(question_df.iterrows(), total=len(question_df)):
        question_count += 1
        try: 
            question = row["text"]
            if MODE == "0":
                ### MODE 0: Original KG_RAG                     ###
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval, node_context_df, CONTEXT_VOLUME, QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD, QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence, model_id=CHAT_MODEL_ID)
                enriched_prompt = "Context: "+ context + "\n" + "Question: "+ question

                # Add delay to avoid API rate limiting
                time.sleep(2)  # 2 second delay between API calls

                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            if MODE == "1":
                ### MODE 1: Structure the retrieval text into JSON format ###
                ### Following PDF specification: Complete JSON restructuring (not incremental) ###
                from kg_rag.rag_based_generation.GPT.mode1_json_restructure import process_mode1
                import json
                import re

                # Get the original context
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval,
                                         node_context_df, CONTEXT_VOLUME,
                                         QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                                         QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence,
                                         model_id=CHAT_MODEL_ID)

                # Process with MODE 1: Complete restructuring to JSON
                structured_json_context = process_mode1(context, question)

                # Create prompt with ONLY the structured JSON (per PDF specification)
                enriched_prompt = "Context (Structured Knowledge Graph):\n" + structured_json_context + "\n\nQuestion: " + question

                # Add delay to avoid API rate limiting
                time.sleep(2)

                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

                # Extract answer from JSON format if the model returns JSON
                try:
                    json_match = re.search(r'\{[^{}]*"answer"\s*:\s*"([^"]+)"[^{}]*\}', output, re.DOTALL)
                    if json_match:
                        output = json_match.group(1)
                    else:
                        # Try to parse as JSON
                        output_cleaned = output.strip()
                        if output_cleaned.startswith('```json'):
                            output_cleaned = output_cleaned[7:]
                        if output_cleaned.endswith('```'):
                            output_cleaned = output_cleaned[:-3]
                        response_json = json.loads(output_cleaned)
                        if isinstance(response_json, dict) and 'answer' in response_json:
                            output = response_json['answer']
                except:
                    # If not JSON or parsing fails, keep original output
                    pass

            if MODE == "2":
                ### MODE 2: Append prior knowledge as suffix from local SPOKE database ###
                ### Following PDF specification: Append some prior knowledge as suffix after the retrieved text ###
                from kg_rag.rag_based_generation.GPT.mode2_prior_knowledge import process_mode2, format_mode2_prompt

                # Get the original context
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval,
                                         node_context_df, CONTEXT_VOLUME,
                                         QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                                         QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence,
                                         model_id=CHAT_MODEL_ID)

                # Process with MODE 2: Append prior knowledge as suffix
                enhanced_context = process_mode2(context, question, node_context_df)

                # Create prompt with enhanced context
                enriched_prompt = format_mode2_prompt(enhanced_context, question)

                # Add delay to avoid API rate limiting
                time.sleep(2)  # 2 second delay between API calls

                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

            if MODE == "3":
                ### MODE 3: Integrate both JSON structure and local SPOKE knowledge ###
                ### Following PDF specification: Integrate these two together ###
                from kg_rag.rag_based_generation.GPT.mode3_integrated import process_mode3, format_mode3_prompt
                import json
                import re

                # Get the original context
                context = retrieve_context(row["text"], vectorstore, embedding_function_for_context_retrieval,
                                         node_context_df, CONTEXT_VOLUME,
                                         QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
                                         QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY, edge_evidence,
                                         model_id=CHAT_MODEL_ID)

                # Process with MODE 3: Integrate JSON structure and prior knowledge
                integrated_context = process_mode3(context, question, node_context_df)

                # Create prompt with integrated context
                enriched_prompt = format_mode3_prompt(integrated_context, question)

                # Add delay to avoid API rate limiting
                time.sleep(2)  # 2 second delay between API calls

                output = get_Gemini_response(enriched_prompt, SYSTEM_PROMPT, temperature=TEMPERATURE)

                # Extract answer from JSON format if present (MODE 3 may include JSON structure)
                try:
                    # Handle various JSON formats the model might return
                    json_match = re.search(r'\{[^{}]*"answer"\s*:\s*"([^"]+)"[^{}]*\}', output, re.DOTALL)
                    if json_match:
                        output = json_match.group(1)
                    else:
                        # Try to parse as JSON
                        output_cleaned = output.strip()
                        if output_cleaned.startswith('```json'):
                            output_cleaned = output_cleaned[7:]
                        if output_cleaned.endswith('```'):
                            output_cleaned = output_cleaned[:-3]
                        response_json = json.loads(output_cleaned)
                        if isinstance(response_json, dict) and 'answer' in response_json:
                            output = response_json['answer']
                except:
                    # If not JSON or parsing fails, keep original output
                    pass

            answer_list.append((row["text"], row["correct_node"], output))

            # Save checkpoint periodically
            if question_count % CHECKPOINT_INTERVAL == 0:
                actual_index = START_INDEX + question_count
                save_checkpoint(answer_list, MODE, actual_index)

        except Exception as e:
            print("Error in processing question: ", row["text"])
            print("Error: ", e)
            answer_list.append((row["text"], row["correct_node"], "Error"))

            # Save checkpoint on error if we've processed enough questions
            if len(answer_list) > 0 and len(answer_list) % 10 == 0:
                actual_index = START_INDEX + len(answer_list)
                save_checkpoint(answer_list, MODE, actual_index)


    answer_df = pd.DataFrame(answer_list, columns=["question", "correct_answer", "llm_answer"])
    output_file = os.path.join(SAVE_PATH, save_name.format(mode=MODE))

    # If we started from a non-zero index, merge with existing results
    if START_INDEX > 0 and os.path.exists(output_file):
        print(f"Merging with existing results (first {START_INDEX} questions)...")
        existing_df = pd.read_csv(output_file)
        # Take first START_INDEX rows from existing, rest from new
        if len(existing_df) >= START_INDEX:
            merged_df = pd.concat([existing_df.iloc[:START_INDEX], answer_df], ignore_index=True)
            answer_df = merged_df
            print(f"Merged: {START_INDEX} existing + {len(answer_list)} new = {len(answer_df)} total")

    answer_df.to_csv(output_file, index=False, header=True)
    print("Save the model outputs in ", output_file)
    print("Completed in {} min".format((time.time()-start_time)/60))

        
        
if __name__ == "__main__":
    main()


