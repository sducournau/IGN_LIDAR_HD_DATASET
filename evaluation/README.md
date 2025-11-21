# IGN LiDAR HD - Evaluation Framework

Comprehensive evaluation system for LiDAR point cloud classification models and AI agents.

## Overview

This evaluation framework provides two complementary evaluation systems:

1. **Classification Evaluation** - Evaluate ML model performance on LiDAR point cloud classification
2. **Agent Evaluation** - Evaluate AI agent performance on natural language LiDAR processing tasks

## Installation

```bash
# Core dependencies (already installed with ign-lidar-hd)
pip install numpy pandas scikit-learn

# For classification evaluation
pip install matplotlib seaborn  # Optional: for visualization

# For AI agent
pip install agent-framework-azure-ai --pre

# For agent evaluation
pip install azure-ai-evaluation

# Set up GitHub token for agent
export GITHUB_TOKEN="your_github_personal_access_token"
# Get token from: https://github.com/settings/tokens
```

## Quick Start

### 1. Classification Model Evaluation

Evaluate your trained classification model's performance:

```python
from evaluation.classification_evaluator import evaluate_classification
import numpy as np

# Load your predictions and ground truth
y_true = np.load("ground_truth_labels.npy")
y_pred = np.load("predicted_labels.npy")
points = np.load("point_coordinates.npy")  # Optional: for spatial metrics

# Run evaluation
metrics = evaluate_classification(
    y_true=y_true,
    y_pred=y_pred,
    points=points,
    output_dir="evaluation_results/classification",
    save_report=True,
)

# View results
print(f"Overall Accuracy: {metrics.overall_accuracy:.2%}")
print(f"Balanced Accuracy: {metrics.balanced_accuracy:.2%}")
print(f"F1 Macro: {metrics.f1_macro:.2%}")
print(f"Kappa: {metrics.kappa_coefficient:.3f}")
```

### 2. AI Agent Evaluation

Evaluate the LiDAR processing AI agent:

```python
from evaluation.agent_evaluator import create_evaluator

# Create evaluator
evaluator = create_evaluator()

# Create test dataset
test_data_path = evaluator.create_test_dataset()

# Run evaluation
results = evaluator.evaluate_agent(test_data_path)

# View results
print(results["metrics"])
```

## Framework Components

### 1. Classification Evaluator (`classification_evaluator.py`)

Evaluates classification model performance with comprehensive metrics:

**Features:**

- Overall accuracy, balanced accuracy, F1 scores
- Per-class metrics (precision, recall, F1, IoU)
- Confusion matrix generation
- Spatial coherence analysis
- Boundary accuracy evaluation
- Error analysis and visualization
- Export to JSON, CSV, and HTML

**Key Classes:**

- `ClassificationEvaluator` - Main evaluator class
- `ClassificationEvaluationMetrics` - Metrics dataclass

**Example Usage:**

```python
from evaluation.classification_evaluator import ClassificationEvaluator
from pathlib import Path

# Initialize evaluator
evaluator = ClassificationEvaluator(
    output_dir=Path("evaluation_results/classification"),
    compute_spatial_metrics=True,
)

# Evaluate single test
metrics = evaluator.evaluate_predictions(
    y_true=ground_truth_labels,
    y_pred=predicted_labels,
    points=point_coordinates,
    confidence=confidence_scores,  # Optional
    test_name="versailles_tiles",
    inference_time_ms=1250.5,  # Optional
)

# Save reports
evaluator.save_report(format="json")
evaluator.save_report(format="html")
evaluator.save_report(format="csv")
```

### 2. LiDAR Agent (`lidar_agent.py`)

AI agent with natural language interface for LiDAR processing:

**Features:**

- Natural language queries about LiDAR data
- Automated processing workflow execution
- Feature computation and classification
- Intelligent parameter recommendations
- Processing status and statistics reporting

**Available Tools:**

- `get_workspace_info()` - Get workspace information
- `list_available_tiles()` - List LAZ/LAS tiles
- `get_classification_schema()` - Get schema information
- `recommend_processing_config()` - Recommend configurations
- `process_tiles()` - Execute processing
- `get_processing_history()` - View processing history
- `analyze_classification_results()` - Analyze results

**Example Usage:**

```python
from evaluation.lidar_agent import create_lidar_agent
import asyncio

async def main():
    # Create agent
    agent = await create_lidar_agent(
        model_id="openai/gpt-4.1-mini",  # GitHub Models
        workspace_dir="/path/to/lidar/data",
    )

    # Interactive chat
    await agent.chat_interactive()

    # Or programmatic usage
    thread = agent.agent.get_new_thread()
    response = await agent.chat(
        "Recommend a configuration for ML training with GPU",
        thread=thread
    )
    print(response)

asyncio.run(main())
```

**Command Line Usage:**

```bash
# Interactive mode
python evaluation/lidar_agent.py

# Demo mode
python evaluation/lidar_agent.py demo
```

### 3. Agent Evaluator (`agent_evaluator.py`)

Evaluates AI agent performance on LiDAR processing tasks:

**Metrics:**

- **Tool Call Correctness** - Verifies correct function calls with proper parameters
- **Response Relevance** - Checks if responses address user questions
- **Processing Recommendations** - Validates appropriate configurations
- **Technical Accuracy** - Ensures correct information about LOD levels, schemas, features

**Custom Evaluators:**

- `ToolCallCorrectnessEvaluator` - Validates tool usage
- `ProcessingRecommendationEvaluator` - Checks configuration appropriateness
- `TechnicalAccuracyEvaluator` - Verifies technical facts

**Example Usage:**

```python
from evaluation.agent_evaluator import create_evaluator
from pathlib import Path

# Create evaluator (uses GitHub Models for RelevanceEvaluator)
evaluator = create_evaluator(
    model_id="gpt-4.1-mini",
    output_dir=Path("evaluation_results/agent"),
)

# Create test dataset
test_data_path = evaluator.create_test_dataset()

# Run evaluation
results = evaluator.evaluate_agent(
    test_data_path=test_data_path,
    output_name="lidar_agent_v1",
)

# View metrics
print("\nEvaluation Metrics:")
for metric, value in results["metrics"].items():
    if isinstance(value, (int, float)):
        print(f"  {metric}: {value:.3f}")
```

## Test Data Format

### Classification Test Data

For classification evaluation, you need:

```python
# Ground truth labels [N] - ASPRS codes
y_true = np.array([2, 2, 6, 6, 11, 3, 4, 5, ...])

# Predicted labels [N] - ASPRS codes
y_pred = np.array([2, 2, 6, 11, 11, 3, 4, 3, ...])

# Optional: Point coordinates [N, 3] for spatial metrics
points = np.array([[x1, y1, z1], [x2, y2, z2], ...])

# Optional: Confidence scores [N]
confidence = np.array([0.95, 0.87, 0.92, ...])
```

### Agent Test Data (JSONL)

For agent evaluation, create a JSONL file with test cases:

```jsonl
{"query": "What classification schemas are available?", "response": "There are ASPRS, LOD2 (15 classes), and LOD3 (30+ classes)...", "tool_calls": [{"name": "get_classification_schema"}], "expected_tools": ["get_classification_schema"], "ground_truth": "ASPRS, LOD2, LOD3"}
{"query": "Recommend config for training", "response": "For training: LOD2, patches_only, GPU enabled...", "tool_calls": [{"name": "recommend_processing_config", "args": {"use_case": "training"}}], "expected_tools": ["recommend_processing_config"], "use_case": "training"}
```

**Required Fields:**

- `query` - User question/request
- `response` - Agent's response
- `tool_calls` - List of tools called (can be empty)
- `expected_tools` - List of expected tool names

**Optional Fields:**

- `use_case` - For recommendation evaluator
- `ground_truth` - Expected response

## Evaluation Outputs

### Classification Evaluation Outputs

```
evaluation_results/classification/
├── evaluation_report.json       # Detailed JSON report
├── evaluation_report.html       # HTML visualization
└── evaluation_report.csv        # CSV summary
```

**JSON Report Structure:**

```json
{
  "evaluation_summary": {
    "num_tests": 1,
    "timestamp": "2025-11-20T10:30:00"
  },
  "tests": [
    {
      "name": "test_versailles",
      "n_points": 1000000,
      "metrics": {
        "overall_accuracy": 0.92,
        "balanced_accuracy": 0.89,
        "f1_macro": 0.88,
        "per_class_precision": {...},
        "per_class_recall": {...},
        "confusion_matrix": [[...]],
        ...
      }
    }
  ]
}
```

### Agent Evaluation Outputs

```
evaluation_results/agent/
├── agent_test_data.jsonl              # Test dataset
├── agent_evaluation_results.json      # Evaluation results
└── agent_evaluation_results/
    ├── instance_results.jsonl         # Per-instance results
    └── evaluation_results.json        # Aggregated metrics
```

**Results Structure:**

```json
{
  "metrics": {
    "tool_call_accuracy": 0.95,
    "recommendation_accuracy": 0.88,
    "technical_accuracy": 0.92,
    "relevance": 0.90
  },
  "rows": [
    {
      "query": "...",
      "response": "...",
      "tool_call_accuracy": 1.0,
      "recommendation_accuracy": 0.9,
      ...
    }
  ]
}
```

## Advanced Usage

### Custom Classification Evaluator

```python
from evaluation.classification_evaluator import ClassificationEvaluator

class CustomEvaluator(ClassificationEvaluator):
    def _compute_custom_metric(self, y_true, y_pred):
        # Your custom metric logic
        return custom_score

    def evaluate_predictions(self, **kwargs):
        metrics = super().evaluate_predictions(**kwargs)
        # Add custom metrics
        metrics.custom_score = self._compute_custom_metric(...)
        return metrics
```

### Custom Agent Evaluator

```python
from evaluation.agent_evaluator import LiDARAgentEvaluator

class AdvancedFeatureEvaluator:
    """Custom evaluator for advanced feature checks."""

    def __init__(self):
        pass

    def __call__(self, *, response: str, **kwargs):
        # Your evaluation logic
        score = self._compute_score(response)
        return {"advanced_feature_score": score}

# Add to evaluator
evaluator = LiDARAgentEvaluator(...)
evaluator.custom_evaluator = AdvancedFeatureEvaluator()
```

### Batch Evaluation

```python
# Evaluate multiple test sets
from evaluation.classification_evaluator import ClassificationEvaluator

evaluator = ClassificationEvaluator(output_dir="batch_eval")

test_sets = [
    ("versailles", versailles_pred, versailles_gt),
    ("paris", paris_pred, paris_gt),
    ("lyon", lyon_pred, lyon_gt),
]

for name, y_pred, y_true in test_sets:
    metrics = evaluator.evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        test_name=name,
    )
    print(f"{name}: Accuracy={metrics.overall_accuracy:.2%}")

# Save combined report
evaluator.save_report(format="html")
```

## Integration with Training Pipeline

### Example: Evaluate During Training

```python
from evaluation.classification_evaluator import evaluate_classification
import torch

def evaluate_epoch(model, val_loader, device):
    """Evaluate model after each epoch."""
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            points, labels = batch
            points, labels = points.to(device), labels.to(device)

            outputs = model(points)
            preds = outputs.argmax(dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # Evaluate
    metrics = evaluate_classification(
        y_true=y_true,
        y_pred=y_pred,
        output_dir=f"training/eval_epoch_{epoch}",
        save_report=True,
    )

    return metrics.overall_accuracy, metrics.f1_macro
```

## Troubleshooting

### Classification Evaluation

**Issue: Memory error with large datasets**

```python
# Process in batches
from evaluation.classification_evaluator import ClassificationEvaluator

evaluator = ClassificationEvaluator(compute_spatial_metrics=False)  # Disable expensive spatial metrics

# Evaluate in chunks
chunk_size = 100000
for i in range(0, len(y_true), chunk_size):
    chunk_true = y_true[i:i+chunk_size]
    chunk_pred = y_pred[i:i+chunk_size]
    metrics = evaluator.evaluate_predictions(chunk_true, chunk_pred, test_name=f"chunk_{i}")
```

### Agent Evaluation

**Issue: Agent Framework not installed**

```bash
pip install agent-framework-azure-ai --pre
```

**Issue: GITHUB_TOKEN not set**

```bash
export GITHUB_TOKEN="ghp_your_token_here"
```

**Issue: Azure AI Evaluation SDK errors**

```bash
pip install --upgrade azure-ai-evaluation
```

## Best Practices

1. **Use consistent class mappings** - Ensure ASPRS codes are consistent between training and evaluation
2. **Include spatial metrics** - Provide point coordinates for boundary accuracy analysis
3. **Save all reports** - Generate JSON, HTML, and CSV for different use cases
4. **Test incrementally** - Start with small test sets, then scale up
5. **Version your evaluations** - Use timestamps or version numbers in output names
6. **Document test scenarios** - Include metadata about test conditions

## Contributing

To add new evaluators:

1. Create evaluator class following the pattern in existing evaluators
2. Add appropriate unit tests
3. Update documentation
4. Submit pull request

## References

- [IGN LiDAR HD Documentation](https://sducournau.github.io/IGN_LIDAR_HD_DATASET/)
- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
- [Azure AI Evaluation SDK](https://learn.microsoft.com/azure/ai-studio/how-to/evaluate-sdk)
- [GitHub Models](https://github.com/marketplace/models)

## License

MIT License - Same as parent IGN LiDAR HD project

## Support

For issues or questions:

- Open an issue on GitHub
- Check the main project documentation
- Review example scripts in `evaluation/examples/`
