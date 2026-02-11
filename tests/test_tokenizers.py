"""
Unit tests for tokenizers.
"""

import pytest
from logoslabs.tokenizers import (
    WhitespaceTokenizer,
    TikTokenTokenizer,
    SentencePieceTokenizer,
    get_tokenizer,
)


class TestWhitespaceTokenizer:
    """Test whitespace tokenizer."""
    
    def test_tokenize_basic(self):
        """Test basic tokenization."""
        tokenizer = WhitespaceTokenizer()
        tokens = tokenizer.tokenize("hello world test")
        assert tokens == ["hello", "world", "test"]
    
    def test_tokenize_empty(self):
        """Test empty string."""
        tokenizer = WhitespaceTokenizer()
        tokens = tokenizer.tokenize("")
        assert tokens == []
    
    def test_tokenize_multiple_spaces(self):
        """Test multiple spaces."""
        tokenizer = WhitespaceTokenizer()
        tokens = tokenizer.tokenize("hello  world   test")
        # split() handles multiple spaces
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens


class TestGetTokenizer:
    """Test tokenizer factory."""
    
    def test_get_whitespace(self):
        """Test getting whitespace tokenizer."""
        tokenizer = get_tokenizer("whitespace")
        assert isinstance(tokenizer, WhitespaceTokenizer)
    
    def test_get_unknown(self):
        """Test unknown tokenizer raises error."""
        with pytest.raises(ValueError, match="Unknown tokenizer"):
            get_tokenizer("unknown")
    
    def test_tiktoken_not_installed(self):
        """Test tiktoken error when not installed."""
        # This will fail if tiktoken IS installed
        try:
            import tiktoken
            pytest.skip("tiktoken is installed, skipping error test")
        except ImportError:
            with pytest.raises(ImportError, match="tiktoken is not installed"):
                get_tokenizer("tiktoken")
    
    def test_sentencepiece_not_installed(self):
        """Test sentencepiece error when not installed."""
        # This will fail if sentencepiece IS installed
        try:
            import sentencepiece
            pytest.skip("sentencepiece is installed, skipping error test")
        except ImportError:
            with pytest.raises(ImportError, match="sentencepiece is not installed"):
                get_tokenizer("sentencepiece")
