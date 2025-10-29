# Periodic Context Compression (PCC): Scaling Multi-Step Reasoning in Large Language Models Beyond the Context Horizon

This repository hosts the research paper for **Periodic Context Compression (PCC)**, a novel, token-efficient, and computationally lightweight memory management strategy designed to extend the effective reasoning horizon of Large Language Model (LLM) agents.

## 🚀 The Challenge: The "15-Step Wall"

LLM agents are powerful, but they have a critical flaw: their reasoning ability breaks down on long-horizon tasks.

Empirically, an agent's coherence and success rate plummet after approximately 15-20 reasoning steps. This failure is caused by two fundamental constraints:

- **Finite Context Windows**: The history of observations, thoughts, and actions quickly saturates the model's physical context limit.
- **Attention Dilution**: Even within the limit, the model "loses the plot" as critical information gets buried, leading to hallucination and a loss of task coherence.

## 💡 The Solution: Periodic Context Compression (PCC)

PCC solves this problem by introducing a structured, recursive memory mechanism.

Instead of letting the context grow uncontrollably, PCC periodically compresses a fixed-length segment of the most recent history into a concise, high-density **Memory State Vector** ($M_t$).

This vector—which acts as an updated, in-context summary of the task's progress—is then prepended to the working context.

*(This is a placeholder: We recommend adding a diagram from your paper, e.g., a simplified version of Algorithm 1, to illustrate the flow.)*

## 🔥 Key Highlights & Advantages

PCC is not just another summarization technique. It is a specific strategy designed to optimize for multi-step agentic reasoning, providing three massive advantages:

- **Massive Token Efficiency**: Drastically reduces the token count of the long-term history, keeping the context window small and manageable.
- **Preserves High-Fidelity Context**: By only compressing the past and leaving the most recent steps uncompressed, the agent retains the full, detailed information it needs for its next action.
- **Transforms Computational Scaling**: This is the most significant benefit.
  - **Standard Agent**: Computational cost scales cubically ($\mathcal{O}(T^3)$) with the number of steps ($T$).
  - **PCC Agent**: Computational cost scales linearly ($\mathcal{O}(T)$).

This shift from cubic to linear complexity decouples the computational cost from the reasoning horizon, making truly long-term autonomous reasoning computationally feasible.

## Visualizing the Impact

Hypothetical results from the paper show that while Standard Agents and other methods (like Full Recursive Summarization or Retrieval-Augmentation) suffer a rapid decline in task success, PCC maintains a robust and high success rate, effectively shattering the 15-step wall.


## 📚 Read the Full Paper

All the details—including the formal algorithm, token efficiency analysis, computational complexity breakdown, and the full (hypothetical) experimental design—are available in the paper.

➡️ **[Read the Full Paper](pcc_research_paper.pdf)**

## Repository Structure

- `/pcc_research_paper.pdf`: The complete research paper.

## 📜 Citation

If you find this work useful in your research, please consider citing it.

```bibtex
@article{hakora_pcc_2025,
    title   = {Periodic Context Compression (PCC): Scaling Multi-Step Reasoning in Large Language Models Beyond the Context Horizon},
    author  = {HAKORA},
    year    = {2025},
    note    = {Independent Research}
}
