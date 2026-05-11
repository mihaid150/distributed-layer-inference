"""
Model splitter utilities for Distributed Layer Inference.

This package downloads or loads a Hugging Face causal language model and
serializes it into stage-specific PyTorch checkpoints such as stage_1.pt,
stage_2.pt, etc.
"""

from dli.model_splitter.splitter import ModelSplitPlan, split_model_into_stages

__all__ = ["ModelSplitPlan", "split_model_into_stages"]
