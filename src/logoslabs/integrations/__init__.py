"""
Integration modules for popular ML/LLM frameworks.

This module provides easy-to-use integrations for:
- LangChain: Output validation and filtering
- LlamaIndex: Query response postprocessing
- ZenML: Pipeline steps and materializers

Usage:
    from logoslabs.integrations.langchain import LogosLabsOutputParser
    from logoslabs.integrations.llamaindex import LogosLabsPostprocessor
    from logoslabs.integrations.zenml import LogosLabsFilterStep
"""

__all__ = []

# Optional imports - only load if frameworks are installed
try:
    from .langchain import LogosLabsOutputParser, LogosLabsChain
    __all__.extend(["LogosLabsOutputParser", "LogosLabsChain"])
except ImportError:
    pass

try:
    from .llamaindex import LogosLabsPostprocessor, LogosLabsNodeFilter
    __all__.extend(["LogosLabsPostprocessor", "LogosLabsNodeFilter"])
except ImportError:
    pass

try:
    from .zenml import LogosLabsFilterStep, LogosLabsScoreMaterializer
    __all__.extend(["LogosLabsFilterStep", "LogosLabsScoreMaterializer"])
except ImportError:
    pass
