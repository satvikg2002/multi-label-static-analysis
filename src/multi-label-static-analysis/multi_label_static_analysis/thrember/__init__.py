"""Vendored EMBER2024 feature extractors used by the research notebooks.

Training and acquisition are orchestrated by the project modules, rather than
upstream wrappers with incompatible feature indices and split assumptions.
Keep features.py unchanged: its digest identifies the validated vector cache.
"""
from .features import (
    ByteEntropyHistogram,
    ByteHistogram,
    GeneralFileInfo,
    StringExtractor,
)

__all__ = [
    "ByteEntropyHistogram",
    "ByteHistogram",
    "GeneralFileInfo",
    "StringExtractor",
]
