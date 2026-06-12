

| File | Stage | What it trains | Reads | Writes |
|---|---|---|---|---|
| `sft_generator.py` | 6.2 | Llama 3.1-8B with QLoRA, learns the citation format | `data/train/sft.jsonl` | `models/sft_generator_lora/` |
| `dpo_generator.py` | 6.3 | DPO on top of SFT, learns to cite faithfully | `data/train/dpo.jsonl` | `models/dpo_generator_lora/` |
| `train_rewriter.py` | 6.4 | Llama 3.2-3B QLoRA for query rewriting | `data/train/rewriter.jsonl` | `models/rewriter_lora/` |
| `train_nli_head.py` | 6.5 | DeBERTa-v3 NLI head, plus temperature calibration | WICE, FEVER and friends | `models/nli_head/` |
| `publish_to_hf.py` | 6.6 | uploads any adapter to HuggingFace | a local adapter folder | a HF repo |


## Why these three models and not the others

The generator is where the citation behavior lives. Off-the-shelf it cites loosely. SFT teaches the format. DPO teaches it to cite the source that actually backs the claim.

The rewriter is small, so adapting it is cheap, and the payoff in retrieval recall is big.

The NLI head needs to be calibrated for the trust math. The backbone we keep frozen, only the small classifier head moves.

The embedder, reranker, and Llama Guard are already strong commodities. Fine-tuning them would not pay off.
