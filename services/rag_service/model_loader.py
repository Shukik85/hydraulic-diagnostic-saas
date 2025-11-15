"""
Ollama-based model loader for RAG service
Simplified replacement for vLLM with local LLM support
"""
import ollama
import structlog
from typing import List, Dict, Any
from pydantic import BaseModel

logger = structlog.get_logger()


class OllamaConfig(BaseModel):
    """Configuration for Ollama LLM"""
    model_name: str = "deepseek-r1:1.5b"  # Lightweight model for CPU
    temperature: float = 0.7
    max_tokens: int = 2048
    host: str = "http://localhost:11434"


class OllamaModelLoader:
    """Load and interact with Ollama models"""
    
    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig()
        self.client = ollama.Client(host=self.config.host)
        self.model_name = self.config.model_name
        logger.info("ollama_initialized", model=self.model_name, host=self.config.host)
    
    def pull_model(self):
        """Pull model from Ollama registry if not exists"""
        try:
            logger.info("pulling_model", model=self.model_name)
            self.client.pull(self.model_name)
            logger.info("model_pulled", model=self.model_name)
        except Exception as e:
            logger.error("model_pull_failed", error=str(e), model=self.model_name)
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Generate text with Ollama"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": temperature or self.config.temperature,
                    "num_predict": max_tokens or self.config.max_tokens,
                }
            )
            
            result = response["message"]["content"]
            logger.info("generation_success", prompt_len=len(prompt), response_len=len(result))
            return result
            
        except Exception as e:
            logger.error("generation_failed", error=str(e))
            raise
    
    def generate_with_reasoning(
        self,
        context: str,
        query: str,
        gnn_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate response with reasoning steps (RAG workflow)"""
        
        system_prompt = """You are an expert hydraulic system diagnostic assistant.
Analyze the provided context, GNN model output, and user query.
Provide:
1. Step-by-step reasoning
2. Clear diagnosis
3. Actionable recommendations

Use Russian language for responses."""
        
        prompt = f"""
Context (Knowledge Base):
{context}

GNN Model Output:
- Anomaly detected: {gnn_output.get('anomaly_detected', False)}
- Confidence: {gnn_output.get('confidence', 0):.2f}
- Component: {gnn_output.get('component_id', 'Unknown')}

User Query: {query}

Provide detailed diagnostic analysis:"""
        
        try:
            response = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3  # Lower for more deterministic diagnostics
            )
            
            return {
                "diagnosis": response,
                "reasoning_steps": self._extract_reasoning_steps(response),
                "confidence": gnn_output.get("confidence", 0),
                "model": self.model_name
            }
        except Exception as e:
            logger.error("reasoning_generation_failed", error=str(e))
            raise
    
    def _extract_reasoning_steps(self, response: str) -> List[str]:
        """Extract reasoning steps from response"""
        # Simple extraction - split by numbered lines
        lines = response.split("\n")
        steps = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-") or line.startswith("•")):
                steps.append(line)
        return steps if steps else [response]
    
    def health_check(self) -> Dict[str, Any]:
        """Check if Ollama is running and model is available"""
        try:
            # List models
            models = self.client.list()
            model_names = [m["name"] for m in models.get("models", [])]
            
            is_ready = self.model_name in model_names
            
            return {
                "status": "healthy" if is_ready else "model_not_loaded",
                "model": self.model_name,
                "available_models": model_names,
                "host": self.config.host
            }
        except Exception as e:
            logger.error("health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.model_name
            }


# Global model instance
_model_instance = None


def get_model() -> OllamaModelLoader:
    """Get or create global model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = OllamaModelLoader()
        try:
            _model_instance.pull_model()
        except Exception as e:
            logger.warning("model_pull_skipped", error=str(e))
    return _model_instance
