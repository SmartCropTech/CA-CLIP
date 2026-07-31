"""Reusable CA-CLIP architecture exported from the training framework."""

try:
    from .ca_clip_framework import (
        Config,
        NativeAlignmentClassifier,
        OpenAIClipEncoder,
        PromptClassAwareResidualGate,
        ResidualAdapter,
    )
except ImportError:
    from ca_clip_framework import (
        Config,
        NativeAlignmentClassifier,
        OpenAIClipEncoder,
        PromptClassAwareResidualGate,
        ResidualAdapter,
    )


CAClipModel = NativeAlignmentClassifier

__all__ = [
    "CAClipModel",
    "Config",
    "NativeAlignmentClassifier",
    "OpenAIClipEncoder",
    "PromptClassAwareResidualGate",
    "ResidualAdapter",
]

