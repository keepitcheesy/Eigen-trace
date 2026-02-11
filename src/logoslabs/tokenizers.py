"""
Tokenizer support for heuristic features.

Provides tokenizer abstraction with:
- Default: whitespace tokenizer (no dependencies)
- Optional: tiktoken and sentencepiece (if installed)
"""

from typing import List, Optional
import sys


class BaseTokenizer:
    """Base tokenizer interface."""
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into a list of tokens."""
        raise NotImplementedError


class WhitespaceTokenizer(BaseTokenizer):
    """Whitespace-based tokenizer (default, no dependencies)."""
    
    def tokenize(self, text: str) -> List[str]:
        """Split text on whitespace."""
        return text.split()


class TikTokenTokenizer(BaseTokenizer):
    """TikToken-based tokenizer (requires tiktoken package)."""
    
    def __init__(self, model: str = "gpt2"):
        """Initialize TikToken tokenizer.
        
        Args:
            model: Model name for tiktoken encoding (default: gpt2)
        """
        try:
            import tiktoken
            self.encoding = tiktoken.get_encoding(model)
        except ImportError:
            raise ImportError(
                "tiktoken is not installed. Please install it with:\n"
                "  pip install tiktoken\n"
                "Or use --tokenizer whitespace (default)"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tiktoken: {e}")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using tiktoken."""
        token_ids = self.encoding.encode(text)
        # Convert token IDs back to string representations
        return [str(tid) for tid in token_ids]


class SentencePieceTokenizer(BaseTokenizer):
    """SentencePiece-based tokenizer (requires sentencepiece package)."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize SentencePiece tokenizer.
        
        Args:
            model_path: Path to SentencePiece model file (optional)
        """
        try:
            import sentencepiece as spm
            if model_path:
                self.sp = spm.SentencePieceProcessor()
                self.sp.load(model_path)
            else:
                # Use a default/mock tokenizer for basic functionality
                # In production, users should provide a model path
                self.sp = None
                print(
                    "Warning: No SentencePiece model provided. "
                    "Using basic whitespace tokenization as fallback.",
                    file=sys.stderr
                )
        except ImportError:
            raise ImportError(
                "sentencepiece is not installed. Please install it with:\n"
                "  pip install sentencepiece\n"
                "Or use --tokenizer whitespace (default)"
            )
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using SentencePiece."""
        if self.sp is None:
            # Fallback to whitespace
            return text.split()
        return self.sp.encode_as_pieces(text)


def get_tokenizer(tokenizer_name: str = "whitespace", **kwargs) -> BaseTokenizer:
    """
    Get a tokenizer by name.
    
    Args:
        tokenizer_name: Name of tokenizer ("whitespace", "tiktoken", "sentencepiece")
        **kwargs: Additional arguments for tokenizer initialization
        
    Returns:
        Tokenizer instance
        
    Raises:
        ValueError: If tokenizer name is not recognized
        ImportError: If required package is not installed
    """
    tokenizer_name = tokenizer_name.lower()
    
    if tokenizer_name == "whitespace":
        return WhitespaceTokenizer()
    elif tokenizer_name == "tiktoken":
        return TikTokenTokenizer(**kwargs)
    elif tokenizer_name == "sentencepiece":
        return SentencePieceTokenizer(**kwargs)
    else:
        raise ValueError(
            f"Unknown tokenizer: {tokenizer_name}. "
            f"Valid options: whitespace, tiktoken, sentencepiece"
        )
