"""Public architecture exports for CA-CLIP."""

try:
    from .ca_clip_framework import (
        CAClipConfig,
        CAClipModel,
        Config,
        NativeAlignmentClassifier,
        OpenAIClipEncoder,
        PromptClassAwareResidualGate,
        ResidualAdapter,
        TextImageEncoder,
    )
except ImportError:
    from ca_clip_framework import (
        CAClipConfig,
        CAClipModel,
        Config,
        NativeAlignmentClassifier,
        OpenAIClipEncoder,
        PromptClassAwareResidualGate,
        ResidualAdapter,
        TextImageEncoder,
    )

__all__ = [
    "CAClipConfig",
    "CAClipModel",
    "Config",
    "NativeAlignmentClassifier",
    "OpenAIClipEncoder",
    "PromptClassAwareResidualGate",
    "ResidualAdapter",
    "TextImageEncoder",
]
