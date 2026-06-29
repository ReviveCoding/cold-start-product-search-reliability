"""Optional Qwen3 embedding and reranking adapters.

These classes are imported only by full/GPU commands. The default smoke pipeline never downloads
external model weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QwenEmbeddingConfig:
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    instruction: str = "Retrieve products that satisfy the shopping query."
    max_length: int = 256
    batch_size: int = 16
    output_dimension: int | None = 512


class QwenEmbedder:
    def __init__(self, config: QwenEmbeddingConfig):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install optional dependencies with `pip install -e .[qwen]` to use Qwen3."
            ) from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.config = config

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        torch = self.torch
        outputs = []
        formatted = (
            [f"Instruct: {self.config.instruction}\nQuery: {text}" for text in texts]
            if is_query
            else texts
        )
        self.model.eval()
        for start in range(0, len(formatted), self.config.batch_size):
            batch = formatted[start : start + self.config.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
            with torch.no_grad():
                hidden = self.model(**encoded).last_hidden_state
                attention_mask = encoded["attention_mask"]
                left_padded = bool(
                    attention_mask[:, -1].sum().item() == attention_mask.shape[0]
                )
                if left_padded:
                    emb = hidden[:, -1]
                else:
                    indices = attention_mask.sum(dim=1) - 1
                    emb = hidden[
                        torch.arange(hidden.size(0), device=hidden.device), indices
                    ]
                if self.config.output_dimension:
                    if not 1 <= self.config.output_dimension <= emb.shape[1]:
                        raise ValueError(
                            f"output_dimension must be in [1, {emb.shape[1]}]"
                        )
                    emb = emb[:, : self.config.output_dimension]
                emb = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
            outputs.append(emb.cpu().numpy())
        return np.vstack(outputs)


@dataclass
class QwenRerankerConfig:
    model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    instruction: str = "Rank products by how well they satisfy the shopping query."
    max_length: int = 1024
    batch_size: int = 8


class QwenReranker:
    """Instruction-aware yes/no reranker following the official Qwen3 scoring interface."""

    def __init__(self, config: QwenRerankerConfig):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Install optional dependencies with `pip install -e .[qwen]` to use Qwen3."
            ) from exc
        self.torch = torch
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        ).eval()
        self.false_token_id = self.tokenizer.convert_tokens_to_ids("no")
        self.true_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix = (
            '<|im_start|>system\nJudge whether the Document meets the requirements based on the '
            'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
            '<|im_end|>\n<|im_start|>user\n'
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        pairs = [
            f"<Instruct>: {self.config.instruction}\n<Query>: {query}\n<Document>: {doc}"
            for doc in documents
        ]
        scores: list[float] = []
        usable_length = self.config.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        for start in range(0, len(pairs), self.config.batch_size):
            batch = pairs[start : start + self.config.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=usable_length,
            )
            encoded["input_ids"] = [
                self.prefix_tokens + ids + self.suffix_tokens for ids in encoded["input_ids"]
            ]
            tensors = self.tokenizer.pad(
                encoded, padding=True, return_tensors="pt", max_length=self.config.max_length
            )
            tensors = {key: value.to(self.model.device) for key, value in tensors.items()}
            with self.torch.no_grad():
                logits = self.model(**tensors).logits[:, -1, :]
                yes_no = self.torch.stack(
                    [logits[:, self.false_token_id], logits[:, self.true_token_id]], dim=1
                )
                probability = self.torch.softmax(yes_no.float(), dim=1)[:, 1]
            scores.extend(probability.cpu().tolist())
        return np.asarray(scores, dtype=float)
