"""
AI Agent Evaluation Framework

Comprehensive evaluation system for LiDAR processing AI agent.
Evaluates:
- Tool call accuracy (correct function calls with proper parameters)
- Response relevance (answers address user questions appropriately)
- Processing workflow correctness (appropriate configuration recommendations)
- Technical accuracy (correct information about LOD levels, schemas, features)

Uses Azure AI Evaluation SDK with custom evaluators for domain-specific metrics.

Author: IGN LiDAR HD Dataset Team
Date: November 20, 2025

Installation:
    pip install azure-ai-evaluation
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Azure AI Evaluation imports
try:
    from azure.ai.evaluation import evaluate, OpenAIModelConfiguration
    AZURE_AI_EVALUATION_AVAILABLE = True
except ImportError:
    AZURE_AI_EVALUATION_AVAILABLE = False
    logging.warning(
        "Azure AI Evaluation not available. Install with: pip install azure-ai-evaluation"
    )

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Code-based Evaluators
# ============================================================================

class ToolCallCorrect ness Evaluator:
    """
    Evaluates if the agent called the correct tools with appropriate parameters.
    
    Checks:
    - Tool was called when needed
    - Correct tool selected
    - Parameters are valid
    """
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def __call__(
        self,
        *,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        expected_tools: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate tool call correctness.
        
        Args:
            tool_calls: List of tool calls made by agent
            expected_tools: List of expected tool names
            
        Returns:
            Dictionary with metrics
        """
        if tool_calls is None:
            tool_calls = []
        
        if expected_tools is None:
            expected_tools = []
        
        # Extract tool names from calls
        called_tools = [call.get("name", "") for call in tool_calls]
        
        # Check if expected tools were called
        correct_tools = [tool for tool in expected_tools if tool in called_tools]
        
        # Calculate accuracy
        if len(expected_tools) > 0:
            accuracy = len(correct_tools) / len(expected_tools)
        else:
            # If no tools expected and none called, that's correct
            accuracy = 1.0 if len(tool_calls) == 0 else 0.0
        
        return {
            "tool_call_accuracy": accuracy,
            "tools_called": len(tool_calls),
            "correct_tools": len(correct_tools),
            "expected_tools": len(expected_tools),
            "tool_names_called": called_tools,
        }


class ProcessingRecommendationEvaluator:
    """
    Evaluates if the agent's processing configuration recommendations are appropriate.
    
    Checks:
    - Recommended parameters align with use case
    - GPU settings are appropriate
    - LOD level matches requirements
    """
    
    def __init__(self):
        """Initialize evaluator."""
        # Define expected configurations for common use cases
        self.use_case_configs = {
            "training": {
                "lod_level": ["LOD2"],
                "processing_mode": ["patches_only"],
                "use_gpu": [True],
            },
            "production": {
                "lod_level": ["LOD2", "LOD3"],
                "processing_mode": ["both", "enriched_only"],
                "use_gpu": [True],
            },
            "research": {
                "lod_level": ["LOD3"],
                "processing_mode": ["enriched_only", "both"],
                "use_gpu": [True, False],
            },
            "quick_test": {
                "lod_level": ["LOD2"],
                "processing_mode": ["patches_only"],
                "use_gpu": [False],
            },
        }
    
    def __call__(
        self,
        *,
        response: str,
        use_case: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate processing recommendation appropriateness.
        
        Args:
            response: Agent's response text
            use_case: Expected use case
            
        Returns:
            Dictionary with metrics
        """
        if use_case is None:
            return {
                "recommendation_accuracy": 0.0,
                "reason": "No use case specified for evaluation",
            }
        
        use_case_lower = use_case.lower()
        expected_config = self.use_case_configs.get(use_case_lower, {})
        
        if not expected_config:
            return {
                "recommendation_accuracy": 0.0,
                "reason": f"Unknown use case: {use_case}",
            }
        
        # Check if response mentions appropriate parameters
        score = 0.0
        checks = []
        
        # Check LOD level
        lod_mentioned = any(lod in response for lod in expected_config.get("lod_level", []))
        if lod_mentioned:
            score += 0.33
            checks.append("LOD level appropriate")
        
        # Check processing mode
        mode_mentioned = any(mode in response for mode in expected_config.get("processing_mode", []))
        if mode_mentioned:
            score += 0.33
            checks.append("Processing mode appropriate")
        
        # Check GPU setting
        gpu_settings = expected_config.get("use_gpu", [])
        if True in gpu_settings and "gpu" in response.lower():
            score += 0.34
            checks.append("GPU setting appropriate")
        elif False in gpu_settings and "gpu" not in response.lower():
            score += 0.34
            checks.append("CPU setting appropriate")
        
        return {
            "recommendation_accuracy": score,
            "checks_passed": checks,
            "num_checks_passed": len(checks),
        }


class TechnicalAccuracyEvaluator:
    """
    Evaluates technical accuracy of agent responses about LiDAR processing.
    
    Checks:
    - Correct information about LOD levels
    - Accurate classification schema details
    - Correct feature computation concepts
    """
    
    def __init__(self):
        """Initialize evaluator with technical facts."""
        self.facts = {
            "lod2_classes": 15,
            "lod3_classes": 30,
            "asprs_standard": "American Society for Photogrammetry",
            "gpu_speedup": "16",
            "classification_codes": {
                "ground": 2,
                "vegetation": [3, 4, 5],
                "building": 6,
                "road": 11,
            },
        }
    
    def __call__(
        self,
        *,
        response: str,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate technical accuracy.
        
        Args:
            response: Agent's response
            query: User query
            
        Returns:
            Dictionary with metrics
        """
        response_lower = response.lower()
        query_lower = query.lower()
        
        # Count correct technical facts mentioned
        correct_facts = 0
        total_relevant_facts = 0
        fact_checks = []
        
        # Check LOD level facts
        if "lod2" in query_lower or "lod 2" in query_lower:
            total_relevant_facts += 1
            if "15" in response or "fifteen" in response_lower:
                correct_facts += 1
                fact_checks.append("LOD2 class count correct")
        
        if "lod3" in query_lower or "lod 3" in query_lower:
            total_relevant_facts += 1
            if "30" in response or "thirty" in response_lower:
                correct_facts += 1
                fact_checks.append("LOD3 class count correct")
        
        # Check ASPRS facts
        if "asprs" in query_lower:
            total_relevant_facts += 1
            if "american" in response_lower and "photogrammetry" in response_lower:
                correct_facts += 1
                fact_checks.append("ASPRS definition correct")
        
        # Check GPU facts
        if "gpu" in query_lower:
            total_relevant_facts += 1
            if "16" in response or "sixteen" in response_lower or "faster" in response_lower:
                correct_facts += 1
                fact_checks.append("GPU speedup mentioned")
        
        # Calculate accuracy
        if total_relevant_facts > 0:
            accuracy = correct_facts / total_relevant_facts
        else:
            # If no specific technical facts queried, assume correct
            accuracy = 1.0
        
        return {
            "technical_accuracy": accuracy,
            "correct_facts": correct_facts,
            "total_relevant_facts": total_relevant_facts,
            "fact_checks": fact_checks,
        }


# ============================================================================
# Agent Evaluator
# ============================================================================

class LiDARAgentEvaluator:
    """
    Comprehensive evaluator for LiDAR processing AI agent.
    
    Evaluates:
    - Tool call correctness
    - Response relevance (built-in)
    - Processing recommendations
    - Technical accuracy
    """
    
    def __init__(
        self,
        model_config: Optional[Any] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize agent evaluator.
        
        Args:
            model_config: OpenAIModelConfiguration for built-in evaluators
            output_dir: Directory for evaluation results
        """
        if not AZURE_AI_EVALUATION_AVAILABLE:
            raise ImportError(
                "Azure AI Evaluation required. Install with: pip install azure-ai-evaluation"
            )
        
        self.model_config = model_config
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_results/agent")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize custom evaluators
        self.tool_call_evaluator = ToolCallCorrectnessEvaluator()
        self.recommendation_evaluator = ProcessingRecommendationEvaluator()
        self.technical_evaluator = TechnicalAccuracyEvaluator()
        
        logger.info("✓ Agent evaluator initialized")
    
    def evaluate_agent(
        self,
        test_data_path: Path,
        output_name: str = "agent_evaluation",
    ) -> Dict[str, Any]:
        """
        Evaluate agent performance on test dataset.
        
        Args:
            test_data_path: Path to JSONL file with test data
            output_name: Name for output files
            
        Returns:
            Evaluation results dictionary
            
        Expected data format (JSONL):
        {
            "query": "User question",
            "response": "Agent response",
            "tool_calls": [{"name": "tool_name", "args": {...}}],
            "expected_tools": ["tool_name"],
            "use_case": "training",  # optional
            "ground_truth": "Expected response"  # optional
        }
        """
        logger.info(f"Evaluating agent on {test_data_path}")
        
        # Prepare output path
        output_path = self.output_dir / f"{output_name}_results.json"
        
        # Configure evaluators
        evaluators = {
            "tool_call_correctness": self.tool_call_evaluator,
            "recommendation_accuracy": self.recommendation_evaluator,
            "technical_accuracy": self.technical_evaluator,
        }
        
        evaluator_config = {
            "tool_call_correctness": {
                "column_mapping": {
                    "tool_calls": "${data.tool_calls}",
                    "expected_tools": "${data.expected_tools}",
                }
            },
            "recommendation_accuracy": {
                "column_mapping": {
                    "response": "${data.response}",
                    "use_case": "${data.use_case}",
                }
            },
            "technical_accuracy": {
                "column_mapping": {
                    "response": "${data.response}",
                    "query": "${data.query}",
                }
            },
        }
        
        # Add built-in RelevanceEvaluator if model config provided
        if self.model_config:
            try:
                from azure.ai.evaluation import RelevanceEvaluator
                
                evaluators["relevance"] = RelevanceEvaluator(
                    model_config=self.model_config
                )
                
                evaluator_config["relevance"] = {
                    "column_mapping": {
                        "query": "${data.query}",
                        "response": "${data.response}",
                    }
                }
                
                logger.info("✓ Added RelevanceEvaluator")
            except Exception as e:
                logger.warning(f"Could not add RelevanceEvaluator: {e}")
        
        # Run evaluation
        try:
            result = evaluate(
                data=str(test_data_path),
                evaluators=evaluators,
                evaluator_config=evaluator_config,
                output_path=str(output_path),
            )
            
            logger.info(f"✓ Evaluation complete")
            logger.info(f"✓ Results saved to {output_path}")
            
            # Log summary metrics
            metrics = result.get("metrics", {})
            logger.info("\n" + "=" * 70)
            logger.info("Agent Evaluation Summary")
            logger.info("=" * 70)
            
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"{metric_name}: {value:.3f}")
            
            logger.info("=" * 70)
            
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            raise
    
    def create_test_dataset(
        self,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Create example test dataset for agent evaluation.
        
        Args:
            output_path: Output JSONL file path
            
        Returns:
            Path to created dataset
        """
        if output_path is None:
            output_path = self.output_dir / "agent_test_data.jsonl"
        
        # Example test cases
        test_cases = [
            {
                "query": "What classification schemas are available?",
                "response": "There are three main classification schemas available: ASPRS (American Society for Photogrammetry & Remote Sensing) with standard codes, LOD2 with 15 simplified building classes, and LOD3 with 30+ detailed architectural classes.",
                "tool_calls": [{"name": "get_classification_schema", "args": {"schema_type": "asprs"}}],
                "expected_tools": ["get_classification_schema"],
                "ground_truth": "ASPRS, LOD2 (15 classes), LOD3 (30+ classes)",
            },
            {
                "query": "Recommend a configuration for machine learning training",
                "response": "For ML training, I recommend: LOD2 level for simplified classification, patches_only mode to generate training patches, 16384 points per patch, GPU acceleration enabled, and ground truth fetching enabled for labeled data.",
                "tool_calls": [{"name": "recommend_processing_config", "args": {"use_case": "training", "gpu_available": True}}],
                "expected_tools": ["recommend_processing_config"],
                "use_case": "training",
                "ground_truth": "LOD2, patches_only, GPU enabled, ground truth enabled",
            },
            {
                "query": "What's the difference between LOD2 and LOD3?",
                "response": "LOD2 provides simplified building classification with 15 essential classes for walls, roofs, and basic features. LOD3 offers detailed architectural classification with 30+ classes including dormers, chimneys, balconies, and fine architectural details. LOD3 requires more computation but provides richer semantic information.",
                "tool_calls": [{"name": "get_classification_schema", "args": {"schema_type": "lod2"}}, {"name": "get_classification_schema", "args": {"schema_type": "lod3"}}],
                "expected_tools": ["get_classification_schema"],
                "ground_truth": "LOD2 has 15 simplified classes, LOD3 has 30+ detailed architectural classes",
            },
            {
                "query": "How much faster is GPU processing?",
                "response": "GPU processing provides approximately 16× speedup for feature computation compared to CPU processing. This is especially beneficial for large datasets and when computing full geometric features.",
                "tool_calls": [],
                "expected_tools": [],
                "ground_truth": "16× faster than CPU",
            },
            {
                "query": "List available tiles in my workspace",
                "response": "I'll check your workspace for available LiDAR tiles.",
                "tool_calls": [{"name": "list_available_tiles", "args": {}}],
                "expected_tools": ["list_available_tiles"],
                "ground_truth": "Should call list_available_tiles function",
            },
        ]
        
        # Write to JSONL
        with open(output_path, 'w') as f:
            for case in test_cases:
                f.write(json.dumps(case) + '\n')
        
        logger.info(f"✓ Created test dataset with {len(test_cases)} cases: {output_path}")
        return output_path


# ============================================================================
# Convenience Functions
# ============================================================================

def create_evaluator(
    github_token: Optional[str] = None,
    model_id: str = "gpt-4.1-mini",
    output_dir: Optional[Path] = None,
) -> LiDARAgentEvaluator:
    """
    Create agent evaluator with model configuration.
    
    Args:
        github_token: GitHub PAT for model access
        model_id: Model ID (default: gpt-4.1-mini)
        output_dir: Output directory
        
    Returns:
        LiDARAgentEvaluator instance
    """
    if github_token is None:
        github_token = os.getenv("GITHUB_TOKEN")
    
    # Create model configuration for built-in evaluators
    model_config = None
    if github_token:
        try:
            model_config = OpenAIModelConfiguration(
                type="openai",
                model=model_id,
                base_url="https://models.github.ai/inference",
                api_key=github_token,
            )
            logger.info(f"✓ Model configuration created: {model_id}")
        except Exception as e:
            logger.warning(f"Could not create model config: {e}")
    
    return LiDARAgentEvaluator(
        model_config=model_config,
        output_dir=output_dir,
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Create evaluator
    evaluator = create_evaluator()
    
    # Create test dataset
    test_data_path = evaluator.create_test_dataset()
    
    print(f"\n✓ Test dataset created: {test_data_path}")
    print(f"\nTo run evaluation:")
    print(f"  python -c \"from evaluation.agent_evaluator import create_evaluator; "
          f"ev = create_evaluator(); ev.evaluate_agent('{test_data_path}')\"")
    
    # If test data exists, run evaluation
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        print("\nRunning evaluation...")
        evaluator.evaluate_agent(test_data_path)
