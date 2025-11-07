"""
MODE 3: Integrate both JSON restructuring and prior knowledge together
Based on PDF specification - Combined enhancement approach
Following classmate's successful strategy with LLM-based conversion
"""
import pandas as pd


def process_mode3(context: str, question: str, node_context_df: pd.DataFrame = None) -> str:
    """
    Main entry point for MODE 3 processing
    Implements: "Integrate these two together" - combining JSON structure and prior knowledge

    According to PDF Figure 2:
    MODE 3 = MODE 1 (JSON structure) + MODE 2 (Prior knowledge)

    Strategy (following classmate's approach):
    1. Use LLM to convert context to JSON (MODE 1)
    2. Append the two specific prior knowledge statements (MODE 2)

    Args:
        context: Retrieved context from vector database
        question: The MCQ question
        node_context_df: SPOKE node context DataFrame (not used in simplified approach)

    Returns:
        Enhanced context with both JSON structure and prior knowledge
    """
    from kg_rag.utility import get_Gemini_response

    # PART 1: JSON Restructuring (MODE 1 - LLM-based conversion)
    # ============================================================
    # Use LLM as intelligent JSON converter (same as MODE 1)
    json_prompt = "Transform the following text into a json object\nText:\n" + context
    json_system_prompt = "You are an expert biomedical researcher."

    # Get JSON-structured context
    structured_json = get_Gemini_response(json_prompt, json_system_prompt, temperature=0.0)

    # Clean markdown code blocks
    if structured_json.startswith("```json\n"):
        structured_json = structured_json.replace("```json\n", "")
    if structured_json.endswith("\n```"):
        structured_json = structured_json.replace("\n```", "")

    # PART 2: Prior Knowledge (MODE 2 - from PDF specification)
    # ===========================================================
    # Add the exact prior knowledge from PDF Figure 2 (yellow highlighted)
    prior_knowledge = """

[Prior Knowledge]:
• Provenance & Symptoms information is useless
• Similar diseases tend to have similar gene associations
"""

    # PART 3: Integration (MODE 3 specific)
    # ======================================
    # Simple concatenation: JSON structure + Prior knowledge
    integrated_output = structured_json + prior_knowledge

    return integrated_output


def format_mode3_prompt(integrated_context: str, question: str) -> str:
    """
    Format the final prompt for MODE 3

    Args:
        integrated_context: Context with both JSON structure and prior knowledge
        question: The MCQ question

    Returns:
        Formatted prompt string
    """
    prompt = f"""Context (JSON-structured with Prior Knowledge): {integrated_context}

Question: {question}

Instructions: Use the structured JSON context and the prior knowledge hints to select the correct answer. Remember that provenance and symptom details may not be as relevant as direct gene-disease associations."""

    return prompt
