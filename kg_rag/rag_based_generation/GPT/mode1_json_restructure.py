"""
MODE 1: Structure the retrieval text into JSON format
Based on PDF specification - Using LLM as intelligent JSON converter
Following classmate's approach for better accuracy and robustness

Key advantages over regex-based approach:
1. Zero information loss - LLM understands natural language semantics
2. Flexible structure - JSON format adapts to content
3. Robust handling - Processes complex medical terminology
"""

def process_mode1(context: str, question: str) -> str:
    """
    Main entry point for MODE 1 processing
    Implements: "Structure the retrieval text into json format"

    Strategy:
    - Use LLM as intelligent JSON converter (not regex hard-parsing)
    - Preserves all semantic information from KG context
    - More robust than pattern matching for complex biomedical text

    Args:
        context: Retrieved KG context in natural language
        question: User question (for reference, not used in this simple approach)

    Returns:
        JSON-structured representation of the context
    """
    from kg_rag.utility import get_Gemini_response

    # Prompt: Direct task instruction with data
    prompt = "Transform the following text into a json object\nText:\n" + context

    # System prompt: Define role identity only (following classmate's proven approach)
    system_prompt = "You are an expert biomedical researcher."

    # Temperature 0.0: Ensures deterministic and consistent output
    resp = get_Gemini_response(prompt, system_prompt, temperature=0.0)

    # Clean markdown code blocks if LLM wraps response in ```json ... ```
    if resp.startswith("```json\n"):
        resp = resp.replace("```json\n", "")
    if resp.endswith("\n```"):
        resp = resp.replace("\n```", "")

    return resp
