"""
LangChain integration for LogosLabs.

Provides output parsing and chain components for validating LLM outputs
using LogosLoss-based instability scoring.

Example:
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    from logoslabs.integrations.langchain import LogosLabsChain
    
    # Create chain with LogosLabs validation
    chain = LogosLabsChain.from_llm(
        llm=OpenAI(),
        threshold=0.5,
        reference_text="Expected output pattern"
    )
    
    result = chain.run("Generate some text")
"""

from typing import Any, Dict, List, Optional, Union
import warnings

try:
    from langchain.schema import BaseOutputParser, OutputParserException
    from langchain.chains import LLMChain
    from langchain.schema.language_model import BaseLanguageModel
    from langchain.prompts import BasePromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    warnings.warn(
        "LangChain is not installed. Install with: pip install langchain"
    )

from ..avp import AVPProcessor


if LANGCHAIN_AVAILABLE:
    class LogosLabsOutputParser(BaseOutputParser[Dict[str, Any]]):
        """
        LangChain output parser that validates LLM outputs using LogosLoss.
        
        This parser computes instability scores for generated text and can
        optionally filter outputs that don't meet quality thresholds.
        
        Args:
            threshold: Instability score threshold (default: 1.0)
            reference_text: Optional reference text for comparison
            grace_coeff: LogosLoss spectral weight (default: 0.5)
            phase_weight: LogosLoss phase weight (default: 0.1)
            raise_on_fail: Raise exception if output fails threshold (default: False)
        """
        
        def __init__(
            self,
            threshold: float = 1.0,
            reference_text: Optional[str] = None,
            grace_coeff: float = 0.5,
            phase_weight: float = 0.1,
            raise_on_fail: bool = False,
        ):
            super().__init__()
            self.threshold = threshold
            self.reference_text = reference_text or ""
            self.raise_on_fail = raise_on_fail
            
            self.processor = AVPProcessor(
                threshold=threshold,
                grace_coeff=grace_coeff,
                phase_weight=phase_weight,
                deterministic=True,
            )
        
        def parse(self, text: str) -> Dict[str, Any]:
            """
            Parse and validate LLM output.
            
            Args:
                text: Generated text to validate
                
            Returns:
                Dictionary with validation results
                
            Raises:
                OutputParserException: If raise_on_fail=True and output fails threshold
            """
            item = {
                "prediction": text,
                "truth": self.reference_text,
            }
            
            result, passed = self.processor.process_item(item)
            
            if self.raise_on_fail and not passed:
                raise OutputParserException(
                    f"Output failed quality threshold. "
                    f"Instability score: {result['instability_score']:.4f} "
                    f"(threshold: {self.threshold})"
                )
            
            return {
                "text": text,
                "instability_score": result["instability_score"],
                "passed_threshold": result["passed_threshold"],
                "validated": passed,
            }
        
        def get_format_instructions(self) -> str:
            """Return format instructions for the parser."""
            return (
                f"Generate high-quality text that will be validated using "
                f"LogosLoss instability scoring (threshold: {self.threshold})."
            )
    
    
    class LogosLabsChain(LLMChain):
        """
        LangChain chain with built-in LogosLabs validation.
        
        This chain automatically validates LLM outputs and can filter or
        flag low-quality responses.
        
        Example:
            chain = LogosLabsChain.from_llm(
                llm=OpenAI(),
                prompt=prompt,
                threshold=0.5,
            )
            result = chain.run(input_text)
        """
        
        threshold: float = 1.0
        reference_text: Optional[str] = None
        grace_coeff: float = 0.5
        phase_weight: float = 0.1
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.processor = AVPProcessor(
                threshold=kwargs.get("threshold", 1.0),
                grace_coeff=kwargs.get("grace_coeff", 0.5),
                phase_weight=kwargs.get("phase_weight", 0.1),
                deterministic=True,
            )
        
        @classmethod
        def from_llm(
            cls,
            llm: BaseLanguageModel,
            prompt: Optional[BasePromptTemplate] = None,
            threshold: float = 1.0,
            reference_text: Optional[str] = None,
            grace_coeff: float = 0.5,
            phase_weight: float = 0.1,
            **kwargs
        ) -> "LogosLabsChain":
            """
            Create LogosLabsChain from LLM and optional prompt.
            
            Args:
                llm: Language model to use
                prompt: Optional prompt template
                threshold: Instability score threshold
                reference_text: Optional reference for comparison
                grace_coeff: LogosLoss spectral weight
                phase_weight: LogosLoss phase weight
                **kwargs: Additional arguments for LLMChain
            
            Returns:
                LogosLabsChain instance
            """
            from langchain.prompts import PromptTemplate
            
            if prompt is None:
                prompt = PromptTemplate(
                    input_variables=["input"],
                    template="{input}"
                )
            
            return cls(
                llm=llm,
                prompt=prompt,
                threshold=threshold,
                reference_text=reference_text,
                grace_coeff=grace_coeff,
                phase_weight=phase_weight,
                **kwargs
            )
        
        def _call(self, inputs: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
            """Call the chain with validation."""
            # Get LLM output
            output = super()._call(inputs, *args, **kwargs)
            
            # Validate output
            text = output.get(self.output_key, "")
            item = {
                "prediction": text,
                "truth": self.reference_text or "",
            }
            
            result, passed = self.processor.process_item(item)
            
            # Add validation metadata
            output["instability_score"] = result["instability_score"]
            output["passed_threshold"] = result["passed_threshold"]
            output["validated"] = passed
            
            return output
else:
    # Dummy classes when LangChain is not installed
    class LogosLabsOutputParser:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LangChain is required for this integration. "
                "Install with: pip install langchain"
            )
    
    class LogosLabsChain:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LangChain is required for this integration. "
                "Install with: pip install langchain"
            )
