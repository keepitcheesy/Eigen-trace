"""
LlamaIndex integration for LogosLabs.

Provides node postprocessors and filters for validating query responses
using LogosLoss-based instability scoring.

Example:
    from llama_index import VectorStoreIndex, SimpleDirectoryReader
    from logoslabs.integrations.llamaindex import LogosLabsPostprocessor
    
    # Create index and query engine
    documents = SimpleDirectoryReader('data').load_data()
    index = VectorStoreIndex.from_documents(documents)
    
    # Add LogosLabs postprocessor
    query_engine = index.as_query_engine(
        node_postprocessors=[
            LogosLabsPostprocessor(threshold=0.5)
        ]
    )
    
    response = query_engine.query("What is...?")
"""

from typing import List, Optional
import warnings

try:
    from llama_index.core.postprocessor import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore, QueryBundle
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    try:
        # Try older import path
        from llama_index.postprocessor import BaseNodePostprocessor
        from llama_index.schema import NodeWithScore, QueryBundle
        LLAMAINDEX_AVAILABLE = True
    except ImportError:
        LLAMAINDEX_AVAILABLE = False
        warnings.warn(
            "LlamaIndex is not installed. Install with: pip install llama-index"
        )

from ..avp import AVPProcessor


if LLAMAINDEX_AVAILABLE:
    class LogosLabsPostprocessor(BaseNodePostprocessor):
        """
        LlamaIndex postprocessor that validates and filters nodes using LogosLoss.
        
        This postprocessor computes instability scores for retrieved nodes
        and can filter out low-quality results.
        
        Args:
            threshold: Instability score threshold (default: 1.0)
            reference_text: Optional reference text for comparison
            grace_coeff: LogosLoss spectral weight (default: 0.5)
            phase_weight: LogosLoss phase weight (default: 0.1)
            filter_failures: Remove nodes that fail threshold (default: False)
        """
        
        def __init__(
            self,
            threshold: float = 1.0,
            reference_text: Optional[str] = None,
            grace_coeff: float = 0.5,
            phase_weight: float = 0.1,
            filter_failures: bool = False,
        ):
            super().__init__()
            self.threshold = threshold
            self.reference_text = reference_text
            self.filter_failures = filter_failures
            
            self.processor = AVPProcessor(
                threshold=threshold,
                grace_coeff=grace_coeff,
                phase_weight=phase_weight,
                deterministic=True,
            )
        
        def _postprocess_nodes(
            self,
            nodes: List[NodeWithScore],
            query_bundle: Optional[QueryBundle] = None,
        ) -> List[NodeWithScore]:
            """
            Postprocess nodes with LogosLabs validation.
            
            Args:
                nodes: List of nodes with scores
                query_bundle: Optional query bundle with query text
                
            Returns:
                Filtered/validated list of nodes
            """
            # Use query text as reference if available
            reference = self.reference_text
            if query_bundle and query_bundle.query_str:
                reference = query_bundle.query_str
            
            validated_nodes = []
            
            for node_with_score in nodes:
                node_text = node_with_score.node.get_content()
                
                # Validate node text
                item = {
                    "prediction": node_text,
                    "truth": reference or "",
                }
                
                result, passed = self.processor.process_item(item)
                
                # Add metadata to node
                node_with_score.node.metadata["instability_score"] = result["instability_score"]
                node_with_score.node.metadata["passed_threshold"] = result["passed_threshold"]
                
                # Filter if requested
                if not self.filter_failures or passed:
                    validated_nodes.append(node_with_score)
            
            return validated_nodes
    
    
    class LogosLabsNodeFilter:
        """
        Simple node filter based on LogosLabs instability scoring.
        
        This can be used as a standalone filter or combined with other
        postprocessors in a LlamaIndex pipeline.
        
        Example:
            filter = LogosLabsNodeFilter(threshold=0.8)
            filtered_nodes = filter.filter_nodes(nodes, query_text)
        """
        
        def __init__(
            self,
            threshold: float = 1.0,
            grace_coeff: float = 0.5,
            phase_weight: float = 0.1,
        ):
            self.processor = AVPProcessor(
                threshold=threshold,
                grace_coeff=grace_coeff,
                phase_weight=phase_weight,
                deterministic=True,
            )
        
        def filter_nodes(
            self,
            nodes: List[NodeWithScore],
            reference_text: str = "",
        ) -> List[NodeWithScore]:
            """
            Filter nodes based on instability scores.
            
            Args:
                nodes: List of nodes to filter
                reference_text: Reference text for comparison
                
            Returns:
                Filtered list of nodes
            """
            filtered = []
            
            for node_with_score in nodes:
                node_text = node_with_score.node.get_content()
                
                item = {
                    "prediction": node_text,
                    "truth": reference_text,
                }
                
                result, passed = self.processor.process_item(item)
                
                if passed:
                    node_with_score.node.metadata["instability_score"] = result["instability_score"]
                    filtered.append(node_with_score)
            
            return filtered

else:
    # Dummy classes when LlamaIndex is not installed
    class LogosLabsPostprocessor:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LlamaIndex is required for this integration. "
                "Install with: pip install llama-index"
            )
    
    class LogosLabsNodeFilter:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "LlamaIndex is required for this integration. "
                "Install with: pip install llama-index"
            )
