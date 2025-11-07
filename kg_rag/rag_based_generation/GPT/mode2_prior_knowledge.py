"""
MODE 2: Append some prior knowledge as suffix after the retrieved text
Based on PDF specification - Prior knowledge enhancement approach
"""
import re
import pandas as pd
from typing import List, Dict, Tuple, Optional

def extract_entities_from_question(question: str) -> Tuple[List[str], List[str]]:
    """
    Extract disease and gene entities from the question text

    Args:
        question: The MCQ question text

    Returns:
        Tuple of (disease_list, gene_list)
    """
    question_diseases = []
    question_genes = []

    # Extract genes from the "Given list" in the question
    gene_list_match = re.search(r'Given list is:\s*([^\.]+)', question)
    if gene_list_match:
        gene_candidates = gene_list_match.group(1).split(',')
        question_genes = [g.strip() for g in gene_candidates if g.strip()]

    # Extract disease names from question using comprehensive keywords
    question_lower = question.lower()

    # Comprehensive disease keywords list
    disease_keywords = [
        # Cancers
        'carcinoma', 'cancer', 'lymphoma', 'leukemia', 'melanoma', 'sarcoma', 'tumor',
        # Autoimmune
        'arthritis', 'lupus', 'psoriasis', 'scleroderma', 'myositis',
        # Neurological
        'alzheimer', 'parkinson', 'epilepsy', 'migraine', 'neuropathy',
        # Metabolic
        'diabetes', 'obesity', 'hyperlipidemia', 'metabolic syndrome',
        # Cardiovascular
        'hypertension', 'atherosclerosis', 'aneurysm', 'stenosis', 'arrhythmia',
        # Respiratory
        'asthma', 'copd', 'fibrosis', 'pneumonia', 'tuberculosis',
        # Liver
        'hepatitis', 'cirrhosis', 'steatosis', 'cholangitis',
        # Kidney
        'glomerulonephritis', 'nephropathy', 'nephritis',
        # Other
        'colitis', 'crohn', 'ulcerative', 'inflammatory bowel',
        'myelodysplastic', 'myeloproliferative', 'takayasu', 'sjogren',
        'hodgkin', 'diffuse large', 'marginal zone', 'uveitis', 'glaucoma',
        'urticaria', 'allergic rhinitis', 'herpes zoster', 'vitiligo',
        'alopecia', 'keratinocyte', 'vasculitis', 'endometrial'
    ]

    # Extract diseases found in question
    for disease in disease_keywords:
        if disease in question_lower:
            # Avoid duplicates and keep the most specific term
            if disease not in [d.lower() for d in question_diseases]:
                question_diseases.append(disease)

    # Also extract specific disease patterns
    disease_patterns = [
        r"which Gene is associated with ([\w\s\-']+) and",
        r"associated with ([\w\s\-']+) and ([\w\s\-']+)",
        r"Variant is associated with ([\w\s\-']+) and",
    ]

    for pattern in disease_patterns:
        matches = re.findall(pattern, question, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                for disease in match:
                    if disease and disease.lower() not in [d.lower() for d in question_diseases]:
                        question_diseases.append(disease.strip())
            else:
                if match and match.lower() not in [d.lower() for d in question_diseases]:
                    question_diseases.append(match.strip())

    return question_diseases, question_genes


def get_prior_knowledge_from_spoke(
    question_diseases: List[str],
    question_genes: List[str],
    node_context_df: pd.DataFrame,
    max_diseases: int = 2,
    max_genes: int = 3
) -> str:
    """
    Retrieve prior knowledge from local SPOKE database

    Args:
        question_diseases: List of diseases extracted from question
        question_genes: List of genes extracted from question
        node_context_df: DataFrame containing SPOKE node context information
        max_diseases: Maximum number of diseases to process
        max_genes: Maximum number of genes to process

    Returns:
        String containing formatted prior knowledge
    """
    prior_knowledge = "\n\n[Additional Prior Knowledge from SPOKE]:\n"
    prior_knowledge += "=" * 50 + "\n"

    # Add gene-specific knowledge
    if question_genes:
        prior_knowledge += "\nGene Associations:\n"
        for gene in question_genes[:max_genes]:
            # Find rows where this gene appears
            gene_rows = node_context_df[
                node_context_df['node_context'].str.contains(
                    f"Gene {gene}\\b", case=False, na=False, regex=True
                )
            ]

            if not gene_rows.empty:
                # Take first relevant context
                additional_context = gene_rows.iloc[0]['node_context']

                # Extract disease associations
                disease_associations = re.findall(
                    r'Disease ([^\.]+?) associates Gene',
                    additional_context
                )

                if disease_associations:
                    # Clean and format associations
                    clean_associations = [d.strip() for d in disease_associations[:5]]
                    prior_knowledge += f"• Gene {gene} is known to associate with: {', '.join(clean_associations)}\n"

                # Extract additional relationships
                relationships = re.findall(
                    r'Gene ' + gene + r' (\w+) (?:Gene|Disease) ([^\.]+)',
                    additional_context
                )
                if relationships and len(relationships) > 0:
                    rel_type, rel_target = relationships[0]
                    prior_knowledge += f"  - {gene} {rel_type} {rel_target}\n"

    # Add disease-specific knowledge
    if question_diseases:
        prior_knowledge += "\nDisease Information:\n"
        for disease in question_diseases[:max_diseases]:
            # Search for disease in node_name or node_context
            disease_rows = node_context_df[
                (node_context_df['node_name'].str.contains(disease, case=False, na=False)) |
                (node_context_df['node_context'].str.contains(f"Disease {disease}", case=False, na=False))
            ]

            if not disease_rows.empty:
                disease_context = disease_rows.iloc[0]['node_context']

                # Extract gene associations
                gene_associations = re.findall(
                    r'associates Gene ([A-Z][A-Z0-9]+)',
                    disease_context
                )

                if gene_associations:
                    unique_genes = list(dict.fromkeys(gene_associations[:8]))  # Remove duplicates, keep order
                    prior_knowledge += f"• {disease.capitalize()} is associated with genes: {', '.join(unique_genes)}\n"

                # Extract disease hierarchy (isa relationships)
                isa_relations = re.findall(
                    r'Disease \w+ isa Disease ([^\.]+?)\.',
                    disease_context
                )

                if isa_relations:
                    unique_relations = list(dict.fromkeys(isa_relations[:3]))
                    prior_knowledge += f"  - {disease.capitalize()} is classified under: {', '.join(unique_relations)}\n"

                # Extract treatment or pathway information if available
                treatment_info = re.findall(
                    r'treats Disease ' + re.escape(disease),
                    disease_context,
                    re.IGNORECASE
                )
                if treatment_info:
                    prior_knowledge += f"  - Treatment information available for {disease}\n"

    # Add relationship summary
    prior_knowledge += "\n" + "=" * 50
    prior_knowledge += "\nNote: This prior knowledge is extracted from the SPOKE biomedical knowledge graph\n"

    return prior_knowledge


def process_mode2(context: str, question: str, node_context_df: pd.DataFrame) -> str:
    """
    Main entry point for MODE 2 processing
    Implements: "Append some prior knowledge as suffix after the retrieved text"

    According to PDF Figure 2 (Yellow highlighted text):
    The prior knowledge to append is EXACTLY:
    1. "Provenance & Symptoms information is useless"
    2. "Similar diseases tend to have similar gene associations"

    These are LITERAL prior knowledge statements, not dynamic extraction instructions.

    Args:
        context: Retrieved context from vector database
        question: The MCQ question
        node_context_df: SPOKE node context DataFrame (not used in corrected implementation)

    Returns:
        Enhanced context with prior knowledge appended as suffix
    """
    # ✅ FIX 6 & 7: Add the TWO exact prior knowledge statements from PDF Figure 2
    # ✅ FIX 8: Remove all SPOKE dynamic extraction logic

    prior_knowledge = """

[Prior Knowledge]:
• Provenance & Symptoms information is useless
• Similar diseases tend to have similar gene associations
"""

    # Append prior knowledge as suffix after retrieved text (PDF requirement)
    enhanced_context = context + prior_knowledge

    return enhanced_context


def format_mode2_prompt(enhanced_context: str, question: str) -> str:
    """
    Format the final prompt for MODE 2

    Args:
        enhanced_context: Context with appended prior knowledge
        question: The MCQ question

    Returns:
        Formatted prompt string
    """
    prompt = f"""Context (with Prior Knowledge): {enhanced_context}

Question: {question}

Instructions: Use the context and the prior knowledge hints to select the correct answer. Remember that provenance and symptom details may not be as relevant as direct gene-disease associations."""

    return prompt