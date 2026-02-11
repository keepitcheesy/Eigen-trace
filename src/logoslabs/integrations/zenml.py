"""
ZenML integration for LogosLabs.

Provides pipeline steps and materializers for integrating LogosLoss-based
validation into ML pipelines.

Example:
    from zenml import pipeline, step
    from logoslabs.integrations.zenml import LogosLabsFilterStep
    
    @pipeline
    def validation_pipeline():
        data = load_data()
        filtered = LogosLabsFilterStep(threshold=0.5)(data)
        return filtered
    
    pipeline_instance = validation_pipeline()
    pipeline_instance.run()
"""

from typing import List, Dict, Any, Optional
import warnings

try:
    from zenml.steps import BaseStep, step, Output
    from zenml.materializers import BaseMaterializer
    ZENML_AVAILABLE = True
except ImportError:
    ZENML_AVAILABLE = False
    warnings.warn(
        "ZenML is not installed. Install with: pip install zenml"
    )

from ..avp import AVPProcessor, load_jsonl, save_jsonl


if ZENML_AVAILABLE:
    @step
    def logoslabs_filter_step(
        items: List[Dict[str, Any]],
        threshold: float = 1.0,
        grace_coeff: float = 0.5,
        phase_weight: float = 0.1,
    ) -> Output(filtered_items=List[Dict[str, Any]], summary=Dict[str, Any]):
        """
        ZenML step for filtering items using LogosLabs.
        
        Args:
            items: List of items with 'prediction' and 'truth' fields
            threshold: Instability score threshold
            grace_coeff: LogosLoss spectral weight
            phase_weight: LogosLoss phase weight
            
        Returns:
            Tuple of (filtered_items, summary)
        """
        processor = AVPProcessor(
            threshold=threshold,
            grace_coeff=grace_coeff,
            phase_weight=phase_weight,
            deterministic=True,
        )
        
        # Process all items
        results = processor.process_batch(items)
        
        # Filter passing items
        filtered = [r for r in results if r["passed_threshold"]]
        
        # Get summary
        summary = processor.get_summary(results)
        
        return filtered, summary
    
    
    class LogosLabsFilterStep(BaseStep):
        """
        Reusable ZenML step for LogosLabs filtering.
        
        This step can be configured and reused across different pipelines.
        
        Example:
            filter_step = LogosLabsFilterStep(
                threshold=0.8,
                grace_coeff=0.5
            )
            filtered_data = filter_step(input_data)
        """
        
        def __init__(
            self,
            threshold: float = 1.0,
            grace_coeff: float = 0.5,
            phase_weight: float = 0.1,
            **kwargs
        ):
            super().__init__(**kwargs)
            self.threshold = threshold
            self.grace_coeff = grace_coeff
            self.phase_weight = phase_weight
        
        def entrypoint(
            self,
            items: List[Dict[str, Any]],
        ) -> Output(filtered_items=List[Dict[str, Any]], summary=Dict[str, Any]):
            """Execute the filtering step."""
            processor = AVPProcessor(
                threshold=self.threshold,
                grace_coeff=self.grace_coeff,
                phase_weight=self.phase_weight,
                deterministic=True,
            )
            
            results = processor.process_batch(items)
            filtered = [r for r in results if r["passed_threshold"]]
            summary = processor.get_summary(results)
            
            return filtered, summary
    
    
    class LogosLabsScoreMaterializer(BaseMaterializer):
        """
        Custom materializer for LogosLabs scoring results.
        
        This materializer handles serialization and deserialization of
        LogosLabs validation results in ZenML pipelines.
        """
        
        ASSOCIATED_TYPES = (list, dict)
        
        def load(self, data_type):
            """Load validation results from artifact store."""
            import json
            from pathlib import Path
            
            file_path = Path(self.uri) / "results.json"
            with open(file_path, "r") as f:
                return json.load(f)
        
        def save(self, data):
            """Save validation results to artifact store."""
            import json
            from pathlib import Path
            
            Path(self.uri).mkdir(parents=True, exist_ok=True)
            file_path = Path(self.uri) / "results.json"
            
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
    
    
    @step
    def logoslabs_score_step(
        items: List[Dict[str, Any]],
        threshold: float = 1.0,
        grace_coeff: float = 0.5,
        phase_weight: float = 0.1,
    ) -> Output(scored_items=List[Dict[str, Any]], summary=Dict[str, Any]):
        """
        ZenML step for scoring items without filtering.
        
        This step adds instability scores to all items but doesn't
        filter them, allowing downstream steps to make filtering decisions.
        
        Args:
            items: List of items with 'prediction' and 'truth' fields
            threshold: Instability score threshold
            grace_coeff: LogosLoss spectral weight
            phase_weight: LogosLoss phase weight
            
        Returns:
            Tuple of (scored_items, summary)
        """
        processor = AVPProcessor(
            threshold=threshold,
            grace_coeff=grace_coeff,
            phase_weight=phase_weight,
            deterministic=True,
        )
        
        results = processor.process_batch(items)
        summary = processor.get_summary(results)
        
        return results, summary

else:
    # Dummy implementations when ZenML is not installed
    def logoslabs_filter_step(*args, **kwargs):
        raise ImportError(
            "ZenML is required for this integration. "
            "Install with: pip install zenml"
        )
    
    class LogosLabsFilterStep:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ZenML is required for this integration. "
                "Install with: pip install zenml"
            )
    
    class LogosLabsScoreMaterializer:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ZenML is required for this integration. "
                "Install with: pip install zenml"
            )
    
    def logoslabs_score_step(*args, **kwargs):
        raise ImportError(
            "ZenML is required for this integration. "
            "Install with: pip install zenml"
        )
