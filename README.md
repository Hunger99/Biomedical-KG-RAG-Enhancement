# KG-RAG (Knowledge Graph-based Retrieval Augmented Generation) in Biomedical Question Answering

**CS 598 JH Course Assignment** - Enhancement of the original [KG-RAG framework](https://github.com/BaranziniLab/KG_RAG) by BaranziniLab.

This project implements and evaluates three enhancement strategies for **KG-RAG** in biomedical question answering using the SPOKE knowledge graph.

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/maszhongming/CS-598-JH-Assignment
cd CS-598-JH-Assignment

# Create conda environment
conda create -n kg_rag python=3.10.9
conda activate kg_rag

# Install dependencies
pip install -r requirements.txt

# Create disease vector database
python -m kg_rag.run_setup
```

### 2. Configure API Key

1. Get your Google API key: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
2. Copy the example config file:
   ```bash
   cp gpt_config.env.example gpt_config.env
   ```
3. **Edit the config file `gpt_config.env`** (located in project root) and add your API key:
   ```bash
   # Open the file
   nano gpt_config.env

   # Replace placeholder with your actual key
   GOOGLE_API_KEY="your-actual-api-key-here"
   ```

**Important**:
- **Config File Location**: `gpt_config.env` in the project root directory
- `gpt_config.env` is in `.gitignore` and will NOT be committed to git
- Never commit your actual API key to the repository
- All modes (0-3) automatically read the API key from `gpt_config.env` via `config.yaml`

---

## Running Experiments

### Execute All Modes

Run each mode (API key is loaded automatically from `gpt_config.env`):

```bash
# MODE 0: Baseline KG-RAG
python -m kg_rag.rag_based_generation.GPT.run_mcq_qa_strategy gemini-2.0-flash 0

# MODE 1: JSON Structured Context
python -m kg_rag.rag_based_generation.GPT.run_mcq_qa_strategy gemini-2.0-flash 1

# MODE 2: Prior Knowledge Enhancement
python -m kg_rag.rag_based_generation.GPT.run_mcq_qa_strategy gemini-2.0-flash 2

# MODE 3: Integrated Approach (MODE 1 + MODE 2)
python -m kg_rag.rag_based_generation.GPT.run_mcq_qa_strategy gemini-2.0-flash 3
```

**Note:** Each mode processes 306 questions and takes 30-50 minutes.

### Resume from Checkpoint (If Interrupted)

If execution stops due to API quota, resume from specific question index:

```bash
# Resume MODE 1 from question 202
python -m kg_rag.rag_based_generation.GPT.run_mcq_qa_strategy gemini-2.0-flash 1 202

# General format: model_name mode_number start_index
python -m kg_rag.rag_based_generation.GPT.run_mcq_qa_strategy [model] [mode] [start_index]
```

**Features:**
- Auto-backup existing results before resuming
- Merges new results with previous questions
- Saves checkpoints every 50 questions

---

## Evaluation & Reports

### Generate Evaluation Reports

After running experiments, generate comprehensive TXT reports:

```bash
python generate_txt_report.py
```

This creates detailed reports in `output/` directory with:
- Summary statistics (accuracy, error count)
- Performance assessment
- Question-by-question analysis
- Incorrect answer summary

### Quick Evaluation (CSV Results)

```bash
# Evaluate specific mode (0, 1, 2, or 3)
python data/my_results/evaluate_gemini.py 0
python data/my_results/evaluate_gemini.py 1
python data/my_results/evaluate_gemini.py 2
python data/my_results/evaluate_gemini.py 3
```

---

## Final Results

### Execution Summary

**Date**: November 7, 2025
**Dataset**: BiomixQA MCQ (306 questions)
**LLM Model**: Gemini-2.0-flash
**Knowledge Graph**: SPOKE (27M+ nodes, 53M+ edges)

### Overall Performance

| Mode | Description | Accuracy | Improvement | Errors |
|------|-------------|----------|-------------|--------|
| **MODE 3** | Integrated (JSON + Prior) | **81.05%** | **+7.19%** ✅ | 0 |
| **MODE 2** | Prior Knowledge | **79.41%** | **+5.55%** | 0 |
| **MODE 1** | LLM-based JSON | **75.16%** | **+1.30%** | 0 |
| **MODE 0** | Baseline KG-RAG | **73.86%** | baseline | 0 |

### Visual Comparison

```
MODE 3 (Integrated):  81.05% ████████████████████████████▌ (+7.19%) ✅ BEST
MODE 2 (Prior):       79.41% ██████████████████████████▌   (+5.55%)
MODE 1 (JSON):        75.16% █████████████████████████     (+1.30%)
MODE 0 (Baseline):    73.86% ████████████████████████▌     baseline
```

### Key Findings

**Success Points** ✅:
1. **MODE 3 Best Performance**: 81.05% accuracy - highest among all modes
2. **All Strategies Improve Baseline**: Even simplest approach (MODE 1) gains +1.30%
3. **Zero Errors**: All modes achieved 100% reliability (no API failures)
4. **Synergy Effect**: MODE 3 (JSON+Prior) outperforms MODE 2 (Prior alone) by +1.64%

**Production Recommendation** ⭐:
- **Use MODE 3** for maximum accuracy (81.05%)
- **Use MODE 2** if minimizing API calls is critical (same calls as baseline, 79.41% accuracy)
- **Avoid MODE 0** - outdated baseline with no enhancements

---

## Implementation Details

### MODE 0 (Baseline)
**File**: Built-in to `run_mcq_qa_strategy.py`

**Strategy**: Direct KG retrieval → Natural language context → LLM

**Pros**: Simple, reliable baseline
**Cons**: No optimization, lowest accuracy (73.86%)

---

### MODE 1 (JSON Structured Context)
**File**: `kg_rag/rag_based_generation/GPT/mode1_json_restructure.py`

**Strategy**: LLM converts natural language KG context to JSON format

**Implementation**:
```python
def process_mode1(context: str, question: str) -> str:
    """Use LLM to convert text context to JSON structure"""
    prompt = "Transform the following text into a json object\nText:\n" + context
    system_prompt = "You are an expert biomedical researcher."
    return get_Gemini_response(prompt, system_prompt, temperature=0.0)
```

**API Calls**: 3 per question
- 1x Entity extraction
- 1x JSON conversion
- 1x Answer generation

**Result**: 75.16% accuracy (+1.30% vs baseline)

**Key Success**: Simplified LLM-based conversion (vs regex parsing) achieves zero errors

---

### MODE 2 (Prior Knowledge Enhancement)
**File**: `kg_rag/rag_based_generation/GPT/mode2_prior_knowledge.py`

**Strategy**: Append domain-specific hints to KG context:
- "Provenance & Symptoms information is useless"
- "Similar diseases tend to have similar gene associations"

**Implementation**:
```python
def process_mode2(context: str, question: str, node_context_df) -> str:
    """Append prior knowledge hints to context"""
    prior_knowledge = """
[Prior Knowledge]:
• Provenance & Symptoms information is useless
• Similar diseases tend to have similar gene associations
"""
    return context + prior_knowledge
```

**API Calls**: 1 per question (same as baseline)

**Result**: 79.41% accuracy (+5.55% vs baseline)

**Key Success**: Simple domain hints significantly boost accuracy with zero overhead

---

### MODE 3 (Integrated Approach)
**File**: `kg_rag/rag_based_generation/GPT/mode3_integrated.py`

**Strategy**: Combine MODE 1 (JSON structure) + MODE 2 (Prior knowledge)

**Implementation**:
```python
def process_mode3(context: str, question: str, node_context_df) -> str:
    """Integrate JSON structure and prior knowledge"""
    # PART 1: Convert to JSON (MODE 1)
    json_prompt = "Transform the following text into a json object\nText:\n" + context
    structured_json = get_Gemini_response(json_prompt, "You are an expert biomedical researcher.", temperature=0.0)

    # PART 2: Add prior knowledge (MODE 2)
    prior_knowledge = """
[Prior Knowledge]:
• Provenance & Symptoms information is useless
• Similar diseases tend to have similar gene associations
"""

    # PART 3: Combine
    return structured_json + prior_knowledge
```

**API Calls**: 3 per question
- 1x Entity extraction
- 1x JSON conversion
- 1x Answer generation

**Result**: 81.05% accuracy (+7.19% vs baseline, +1.64% vs MODE 2)

**Key Success**: Synergy between structured representation and domain hints achieves best performance

---

## Recommendations

### For Production Use

**Recommended: MODE 3 (Integrated)** ⭐
- **Highest accuracy**: 81.05%
- **Zero errors**: 100% reliability
- **Best overall performance**: +7.19% vs baseline
- **Trade-off**: 3x API calls vs baseline (acceptable for quality gain)

**Alternative: MODE 2 (Prior Knowledge)**
- **High accuracy**: 79.41% (only -1.64% vs MODE 3)
- **Minimal overhead**: Same API calls as baseline
- **Best cost-performance ratio**
- **Use when**: API quota is limited

**Avoid: MODE 0 (Baseline)**
- Outdated with no enhancements
- Lowest accuracy (73.86%)

### For Future Improvements

1. **Test additional domain hints**: Expand prior knowledge statements
2. **Optimize JSON schema**: Experiment with different structured formats
3. **Hybrid retrieval**: Combine vector search with graph traversal
4. **Fine-tune prompts**: A/B test prompt variations for MODE 3

---

## Project Structure

```
CS-598-JH-Assignment-main/
├── output/                           # Evaluation reports (TXT format)
│   ├── mode_0_report.txt            # Baseline results (73.86%)
│   ├── mode_1_report.txt            # MODE 1 results (75.16%)
│   ├── mode_2_report.txt            # MODE 2 results (79.41%)
│   └── mode_3_report.txt            # MODE 3 results (81.05%)
├── data/
│   ├── my_results/                  # Raw CSV results
│   │   ├── gemini_2.0_flash_kg_rag_based_mcq_0_strategy.csv
│   │   ├── gemini_2.0_flash_kg_rag_based_mcq_1_strategy.csv
│   │   ├── gemini_2.0_flash_kg_rag_based_mcq_2_strategy.csv
│   │   └── gemini_2.0_flash_kg_rag_based_mcq_3_strategy.csv
│   └── benchmark_data/              # Test questions (MCQ)
├── kg_rag/
│   └── rag_based_generation/GPT/
│       ├── run_mcq_qa_strategy.py   # Main execution script
│       ├── mode1_json_restructure.py # MODE 1 implementation
│       ├── mode2_prior_knowledge.py  # MODE 2 implementation
│       └── mode3_integrated.py       # MODE 3 implementation
├── generate_txt_report.py           # Report generation script
├── cleanup_for_github.sh            # GitHub preparation script
└── config.yaml                      # Configuration parameters
```

---

## About KG-RAG

**KG-RAG** combines explicit knowledge from Knowledge Graphs with implicit knowledge from Large Language Models. It extracts "prompt-aware context" from the SPOKE biomedical knowledge graph - the minimal context sufficient to respond to user prompts.

**SPOKE KG Features:**
- 27M+ nodes (21 types)
- 53M+ edges (55 types)
- Integrates 40+ biomedical repositories
- Covers genes, proteins, drugs, diseases, and their relationships

**Reference:** [arXiv:2311.17330](https://arxiv.org/abs/2311.17330)

---

## Configuration

Key parameters in `config.yaml`:

```yaml
CONTEXT_VOLUME: 150                    # Number of context sentences
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD: 75
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY: 0.5
LLM_TEMPERATURE: 0                     # For reproducibility
```

---

## Troubleshooting

**API Rate Limiting:**
- Gemini-2.0-flash has daily quota limits
- Use resume functionality to continue from last checkpoint
- Checkpoints saved every 50 questions automatically

**Memory Issues:**
- Reduce `CONTEXT_VOLUME` if running out of memory
- Process questions in batches using start_index parameter

**Missing Dependencies:**
- Ensure all packages in `requirements.txt` are installed
- Check Python version is 3.10.9

---

## Reference & Acknowledgments

This project is based on the original **KG-RAG** framework developed by BaranziniLab:
- **Original Repository**: [https://github.com/BaranziniLab/KG_RAG](https://github.com/BaranziniLab/KG_RAG)
- **Paper**: [arXiv:2311.17330](https://arxiv.org/abs/2311.17330)

**Course**: CS 598 JH - Knowledge Graphs and Large Language Models
**Knowledge Graph**: SPOKE (Scalable Precision Medicine Oriented Knowledge Engine)
**Dataset**: BiomixQA - Biomedical Multiple Choice Question Answering

---

## License

See [LICENSE](LICENSE) file for details.

---

**Last Updated**: 2025-11-07
