"""
LiDAR Processing AI Agent

Natural language interface for IGN LiDAR HD processing operations.
Uses Microsoft Agent Framework with GitHub Models.

Features:
- Natural language queries about LiDAR data
- Automated processing workflow execution
- Feature computation and classification
- Intelligent parameter recommendations
- Processing status and statistics reporting

Author: IGN LiDAR HD Dataset Team  
Date: November 20, 2025

Installation:
    pip install agent-framework-azure-ai --pre

Setup:
    export GITHUB_TOKEN="your_github_pat_token"
    
    Or get token from: https://github.com/settings/tokens
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Annotated
import numpy as np
from datetime import datetime

# Agent Framework imports
try:
    from agent_framework import ChatAgent
    from agent_framework.openai import OpenAIChatClient
    from openai import AsyncOpenAI
    AGENT_FRAMEWORK_AVAILABLE = True
except ImportError:
    AGENT_FRAMEWORK_AVAILABLE = False
    logging.warning(
        "Agent Framework not available. Install with: pip install agent-framework-azure-ai --pre"
    )

# Import LiDAR processing modules
from ign_lidar import LiDARProcessor, Config
from ign_lidar.classification_schema import ASPRS_CLASS_NAMES

logger = logging.getLogger(__name__)


# ============================================================================
# LiDAR Agent Tools
# ============================================================================

class LiDARAgentTools:
    """Tools for LiDAR processing agent."""
    
    def __init__(self, workspace_dir: Optional[Path] = None):
        """
        Initialize agent tools.
        
        Args:
            workspace_dir: Working directory for LiDAR data
        """
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path(".")
        self.processor: Optional[LiDARProcessor] = None
        self.processing_history: List[Dict[str, Any]] = []
    
    def get_workspace_info(self) -> str:
        """Get information about current workspace."""
        info = {
            "workspace_directory": str(self.workspace_dir.absolute()),
            "tiles_available": self._count_tiles(),
            "processed_outputs": self._count_outputs(),
        }
        return json.dumps(info, indent=2)
    
    def _count_tiles(self) -> int:
        """Count available LAZ/LAS tiles."""
        tiles = list(self.workspace_dir.glob("**/*.la[sz]"))
        return len(tiles)
    
    def _count_outputs(self) -> int:
        """Count processed output files."""
        outputs = list(self.workspace_dir.glob("**/patches/*.npz"))
        return len(outputs)
    
    def list_available_tiles(
        self,
        directory: Annotated[Optional[str], "Directory to search for tiles"] = None
    ) -> str:
        """
        List all available LiDAR tiles in the workspace.
        
        Args:
            directory: Optional specific directory to search
            
        Returns:
            JSON string with tile information
        """
        search_dir = Path(directory) if directory else self.workspace_dir
        tiles = list(search_dir.glob("**/*.la[sz]"))
        
        tile_info = []
        for tile_path in tiles[:20]:  # Limit to 20 tiles
            size_mb = tile_path.stat().st_size / (1024 * 1024)
            tile_info.append({
                "name": tile_path.name,
                "path": str(tile_path),
                "size_mb": round(size_mb, 2),
            })
        
        result = {
            "total_tiles": len(tiles),
            "showing": min(20, len(tiles)),
            "tiles": tile_info,
        }
        
        return json.dumps(result, indent=2)
    
    def get_classification_schema(
        self,
        schema_type: Annotated[str, "Schema type: 'asprs', 'lod2', or 'lod3'"] = "asprs"
    ) -> str:
        """
        Get information about classification schemas.
        
        Args:
            schema_type: Type of schema to describe
            
        Returns:
            JSON string with schema information
        """
        from ign_lidar.classification_schema import LOD2_CLASSES, LOD3_CLASSES
        
        if schema_type.lower() == "asprs":
            schema = {name: int(code) for code, name in ASPRS_CLASS_NAMES.items()}
            description = "ASPRS Standard Classification Codes"
        elif schema_type.lower() == "lod2":
            schema = {cls.name: cls.value for cls in LOD2_CLASSES}
            description = "Level of Detail 2 (LOD2) - Simplified Building Classification"
        elif schema_type.lower() == "lod3":
            schema = {cls.name: cls.value for cls in LOD3_CLASSES}
            description = "Level of Detail 3 (LOD3) - Detailed Architectural Classification"
        else:
            return json.dumps({"error": f"Unknown schema type: {schema_type}"})
        
        return json.dumps({
            "schema_type": schema_type.upper(),
            "description": description,
            "classes": schema,
            "num_classes": len(schema),
        }, indent=2)
    
    def recommend_processing_config(
        self,
        use_case: Annotated[str, "Use case: 'training', 'production', 'research', 'quick_test'"],
        gpu_available: Annotated[bool, "Whether GPU is available"] = True,
    ) -> str:
        """
        Recommend processing configuration based on use case.
        
        Args:
            use_case: Intended use case
            gpu_available: Whether GPU acceleration is available
            
        Returns:
            JSON string with recommended configuration
        """
        configs = {
            "training": {
                "description": "Optimized for ML model training",
                "lod_level": "LOD2",
                "processing_mode": "patches_only",
                "patch_size": 150.0,
                "num_points": 16384,
                "use_gpu": gpu_available,
                "feature_mode": "lod2",
                "ground_truth_enabled": True,
            },
            "production": {
                "description": "High-quality production processing",
                "lod_level": "LOD3",
                "processing_mode": "both",
                "patch_size": 200.0,
                "num_points": 32768,
                "use_gpu": gpu_available,
                "feature_mode": "full",
                "ground_truth_enabled": True,
            },
            "research": {
                "description": "Comprehensive feature extraction for research",
                "lod_level": "LOD3",
                "processing_mode": "enriched_only",
                "use_gpu": gpu_available,
                "feature_mode": "full",
                "ground_truth_enabled": False,
            },
            "quick_test": {
                "description": "Fast processing for testing",
                "lod_level": "LOD2",
                "processing_mode": "patches_only",
                "patch_size": 100.0,
                "num_points": 8192,
                "use_gpu": False,
                "feature_mode": "minimal",
                "ground_truth_enabled": False,
            },
        }
        
        config = configs.get(use_case.lower())
        if not config:
            return json.dumps({
                "error": f"Unknown use case: {use_case}",
                "available_use_cases": list(configs.keys()),
            })
        
        return json.dumps({
            "use_case": use_case,
            "recommended_config": config,
            "note": "Adjust parameters based on your specific needs and hardware",
        }, indent=2)
    
    def process_tiles(
        self,
        input_dir: Annotated[str, "Directory containing LAZ/LAS tiles"],
        output_dir: Annotated[str, "Output directory for processed data"],
        lod_level: Annotated[str, "LOD level: 'LOD2' or 'LOD3'"] = "LOD2",
        use_gpu: Annotated[bool, "Use GPU acceleration"] = True,
    ) -> str:
        """
        Process LiDAR tiles with specified configuration.
        
        Args:
            input_dir: Input directory with tiles
            output_dir: Output directory
            lod_level: Level of detail
            use_gpu: Use GPU acceleration
            
        Returns:
            JSON string with processing results
        """
        try:
            # Create config
            config = Config(
                input_dir=input_dir,
                output_dir=output_dir,
                lod_level=lod_level,
                use_gpu=use_gpu,
            )
            
            # Create processor
            self.processor = LiDARProcessor(config=config)
            
            # Start processing
            logger.info(f"Starting LiDAR processing: {input_dir} -> {output_dir}")
            start_time = datetime.now()
            
            # Process (this is synchronous - in production, run in separate thread)
            stats = self.processor.process()
            
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            # Record history
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "input_dir": input_dir,
                "output_dir": output_dir,
                "lod_level": lod_level,
                "use_gpu": use_gpu,
                "elapsed_time_seconds": elapsed_time,
                "stats": stats,
            }
            self.processing_history.append(history_entry)
            
            return json.dumps({
                "status": "success",
                "elapsed_time_seconds": elapsed_time,
                "statistics": stats,
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return json.dumps({
                "status": "error",
                "error": str(e),
            })
    
    def get_processing_history(self) -> str:
        """Get history of processing operations."""
        return json.dumps({
            "total_operations": len(self.processing_history),
            "history": self.processing_history[-10:],  # Last 10 operations
        }, indent=2)
    
    def analyze_classification_results(
        self,
        results_file: Annotated[str, "Path to NPZ file with classification results"]
    ) -> str:
        """
        Analyze classification results from processed data.
        
        Args:
            results_file: Path to NPZ results file
            
        Returns:
            JSON string with analysis
        """
        try:
            data = np.load(results_file)
            labels = data['labels']
            
            # Count classes
            unique, counts = np.unique(labels, return_counts=True)
            
            # Build class distribution
            distribution = {}
            for class_id, count in zip(unique, counts):
                class_name = ASPRS_CLASS_NAMES.get(class_id, f"Unknown_{class_id}")
                distribution[class_name] = {
                    "count": int(count),
                    "percentage": float(count / len(labels) * 100),
                }
            
            return json.dumps({
                "file": results_file,
                "total_points": len(labels),
                "num_classes": len(unique),
                "class_distribution": distribution,
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Failed to analyze results: {e}"
            })


# ============================================================================
# LiDAR Agent
# ============================================================================

class LiDARAgent:
    """
    AI Agent for LiDAR processing with natural language interface.
    
    Features:
    - Answer questions about LiDAR processing
    - Execute processing workflows
    - Provide configuration recommendations
    - Analyze results and statistics
    
    Requires:
    - GitHub Personal Access Token (PAT)
    - agent-framework-azure-ai package
    """
    
    def __init__(
        self,
        github_token: str,
        model_id: str = "openai/gpt-4.1-mini",
        workspace_dir: Optional[Path] = None,
    ):
        """
        Initialize LiDAR agent.
        
        Args:
            github_token: GitHub Personal Access Token
            model_id: Model ID from GitHub Models (default: gpt-4.1-mini)
            workspace_dir: Working directory for LiDAR operations
        """
        if not AGENT_FRAMEWORK_AVAILABLE:
            raise ImportError(
                "Agent Framework required. Install with: pip install agent-framework-azure-ai --pre"
            )
        
        self.github_token = github_token
        self.model_id = model_id
        self.workspace_dir = workspace_dir
        
        # Initialize tools
        self.tools = LiDARAgentTools(workspace_dir=workspace_dir)
        
        # Create OpenAI client
        self.openai_client = AsyncOpenAI(
            base_url="https://models.github.ai/inference",
            api_key=github_token,
        )
        
        # Create chat client
        self.chat_client = OpenAIChatClient(
            async_client=self.openai_client,
            model_id=model_id,
        )
        
        # Agent instructions
        instructions = """You are an expert AI assistant for IGN LiDAR HD point cloud processing.

You help users with:
- Understanding LiDAR data processing workflows
- Configuring and running processing pipelines
- Analyzing classification results
- Recommending optimal parameters
- Troubleshooting processing issues

Key concepts:
- LOD2: Simplified building classification (15 classes)
- LOD3: Detailed architectural classification (30+ classes)
- ASPRS: American Society for Photogrammetry & Remote Sensing standard
- GPU acceleration: 16× faster feature computation
- Ground truth: BD TOPO building/road data from IGN

Always provide clear, actionable advice. When processing data, explain what's happening and why."""
        
        # Create agent with tools
        self.agent = ChatAgent(
            chat_client=self.chat_client,
            name="LiDARAgent",
            instructions=instructions,
            tools=[
                self.tools.get_workspace_info,
                self.tools.list_available_tiles,
                self.tools.get_classification_schema,
                self.tools.recommend_processing_config,
                self.tools.process_tiles,
                self.tools.get_processing_history,
                self.tools.analyze_classification_results,
            ],
        )
        
        logger.info(f"✓ LiDAR Agent initialized with model: {model_id}")
    
    async def chat(self, message: str, thread=None) -> str:
        """
        Send a message to the agent and get response.
        
        Args:
            message: User message
            thread: Optional thread for multi-turn conversation
            
        Returns:
            Agent response text
        """
        response_text = ""
        
        async for chunk in self.agent.run_stream(message, thread=thread):
            if chunk.text:
                response_text += chunk.text
        
        return response_text
    
    async def chat_interactive(self):
        """Run interactive chat session."""
        print("=" * 70)
        print("IGN LiDAR HD Processing Agent")
        print("=" * 70)
        print(f"Model: {self.model_id}")
        print(f"Workspace: {self.workspace_dir or 'current directory'}")
        print("\nType 'quit' or 'exit' to end session\n")
        
        # Create thread for conversation
        thread = self.agent.get_new_thread()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\nGoodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("\nAgent: ", end="", flush=True)
                
                async for chunk in self.agent.run_stream(user_input, thread=thread):
                    if chunk.text:
                        print(chunk.text, end="", flush=True)
                
                print("\n")
                
            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n\nError: {e}")
                logger.error(f"Chat error: {e}", exc_info=True)


# ============================================================================
# Convenience Functions
# ============================================================================

async def create_lidar_agent(
    github_token: Optional[str] = None,
    model_id: str = "openai/gpt-4.1-mini",
    workspace_dir: Optional[Path] = None,
) -> LiDARAgent:
    """
    Create and initialize LiDAR agent.
    
    Args:
        github_token: GitHub PAT (reads from GITHUB_TOKEN env if None)
        model_id: Model ID from GitHub Models
        workspace_dir: Working directory
        
    Returns:
        Initialized LiDARAgent
    """
    import os
    
    if github_token is None:
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise ValueError(
                "GitHub token required. Set GITHUB_TOKEN environment variable or pass explicitly.\n"
                "Get token from: https://github.com/settings/tokens"
            )
    
    return LiDARAgent(
        github_token=github_token,
        model_id=model_id,
        workspace_dir=workspace_dir,
    )


async def demo_agent():
    """Demo the agent with example queries."""
    agent = await create_lidar_agent()
    thread = agent.agent.get_new_thread()
    
    print("=" * 70)
    print("LiDAR Agent Demo")
    print("=" * 70)
    print()
    
    queries = [
        "What classification schemas are available?",
        "What's the difference between LOD2 and LOD3?",
        "Recommend a configuration for ML model training with GPU",
        "List available tiles in the workspace",
    ]
    
    for query in queries:
        print(f"User: {query}")
        print("Agent: ", end="", flush=True)
        
        async for chunk in agent.agent.run_stream(query, thread=thread):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        
        print("\n" + "-" * 70 + "\n")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Example usage
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        asyncio.run(demo_agent())
    else:
        # Interactive mode
        async def main():
            agent = await create_lidar_agent()
            await agent.chat_interactive()
        
        asyncio.run(main())
