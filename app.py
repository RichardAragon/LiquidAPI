import torch
import json
import logging
import asyncio
import aiohttp
import yaml
import os
from typing import Dict, Any, Optional, Type
from dataclasses import dataclass
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from abc import ABC, abstractmethod
from aiohttp import web

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("liquid_api.log")
    ]
)
logger = logging.getLogger('LiquidAPI')

@dataclass
class APIConfig:
    """Configuration for an API endpoint"""
    url: str
    method: str = 'GET'
    headers: Optional[Dict[str, str]] = None
    auth: Optional[Dict[str, str]] = None
    retry_attempts: int = 3
    timeout: int = 30

class LLMBrain:
    """Advanced LLM-based decision making component"""
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-135M",
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing LLMBrain with model {model_name} on {self.device}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    async def analyze_context(
        self,
        context: Dict[str, Any],
        task_description: str
    ) -> Dict[str, Any]:
        """Analyze context and make intelligent decisions"""
        prompt = self._construct_prompt(context, task_description)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=500,
                temperature=0.7,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            decision_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            # Attempt to parse the decision as JSON
            try:
                decision = json.loads(decision_text)
                confidence = torch.mean(torch.stack([torch.max(score) for score in outputs.scores])).item()
            except json.JSONDecodeError:
                logger.error("LLM decision is not in valid JSON format.")
                decision = {"action_to_take": "default"}
                confidence = 0.0
            
            return {
                "decision": decision,
                "confidence": confidence,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            raise

    def _construct_prompt(self, context: Dict[str, Any], task_description: str) -> str:
        return f"""As an API integration coordinator, analyze this context and provide a solution in JSON format:

Context:
{json.dumps(context, indent=2)}

Task:
{task_description}

Provide a structured JSON response with the following fields:
1. action_to_take: Describe the action to take.
2. required_transformations: List any data transformations required.
3. error_handling: Describe error handling considerations.
4. success_criteria: Define success criteria."""

class APIAdapter(ABC):
    """Abstract base class for API adapters"""
    def __init__(self, config: APIConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers=self.config.headers,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @abstractmethod
    async def transform_request(self, data: Any) -> Any:
        """Transform request data before sending"""
        pass

    @abstractmethod
    async def transform_response(self, response: Any) -> Any:
        """Transform response data after receiving"""
        pass

    async def execute(self, data: Any = None) -> Any:
        """Execute the API call with retry logic"""
        for attempt in range(self.config.retry_attempts):
            try:
                transformed_data = await self.transform_request(data)
                async with self.session.request(
                    self.config.method,
                    self.config.url,
                    json=transformed_data if data else None,
                    auth=aiohttp.BasicAuth(**self.config.auth) if self.config.auth else None
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return await self.transform_response(result)
            except Exception as e:
                logger.error(f"API call attempt {attempt + 1} failed: {e}")
                if attempt == self.config.retry_attempts - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

# Example concrete adapter implementations

class CRMAdapter(APIAdapter):
    """Concrete APIAdapter for CRM system"""
    async def transform_request(self, data: Any) -> Any:
        # Example transformation: map internal fields to CRM API fields
        if data:
            transformed = {
                "customer_number": data.get("erp_customer_number"),
                "order_info": data.get("sales_order")
            }
            logger.debug(f"CRMAdapter transformed request: {transformed}")
            return transformed
        return {}

    async def transform_response(self, response: Any) -> Any:
        # Example transformation: extract relevant information from CRM response
        transformed = {
            "crm_status": response.get("status"),
            "crm_message": response.get("message")
        }
        logger.debug(f"CRMAdapter transformed response: {transformed}")
        return transformed

class ERPAdapter(APIAdapter):
    """Concrete APIAdapter for ERP system"""
    async def transform_request(self, data: Any) -> Any:
        # Example transformation: map internal fields to ERP API fields
        if data:
            transformed = {
                "erp_customer_id": data.get("customer_id"),
                "erp_order_details": data.get("order_details")
            }
            logger.debug(f"ERPAdapter transformed request: {transformed}")
            return transformed
        return {}

    async def transform_response(self, response: Any) -> Any:
        # Example transformation: extract relevant information from ERP response
        transformed = {
            "erp_status": response.get("status"),
            "erp_confirmation": response.get("confirmation_number")
        }
        logger.debug(f"ERPAdapter transformed response: {transformed}")
        return transformed

class GenericAPIAdapter(APIAdapter):
    """A generic adapter for APIs that do not require special transformations"""
    async def transform_request(self, data: Any) -> Any:
        # No transformation
        logger.debug(f"GenericAPIAdapter passing through request data: {data}")
        return data

    async def transform_response(self, response: Any) -> Any:
        # No transformation
        logger.debug(f"GenericAPIAdapter passing through response data: {response}")
        return response

class LiquidAPI:
    """Main orchestrator for intelligent API integration"""
    adapter_registry: Dict[str, Type[APIAdapter]] = {
        "crm_adapter": CRMAdapter,
        "erp_adapter": ERPAdapter,
        "generic_adapter": GenericAPIAdapter
        # Add more adapters here as needed
    }

    def __init__(self, brain: LLMBrain):
        self.brain = brain
        self.adapters: Dict[str, APIAdapter] = {}

    def register_adapter(self, name: str, adapter: APIAdapter):
        """Register a new API adapter"""
        self.adapters[name] = adapter
        logger.info(f"Registered adapter: {name}")

    async def integrate(
        self,
        source_adapter: str,
        target_adapter: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Orchestrate integration between two APIs based on LLM decisions"""
        if source_adapter not in self.adapters or target_adapter not in self.adapters:
            raise ValueError("Invalid adapter specified")

        # Analyze the integration context
        analysis = await self.brain.analyze_context(
            context,
            f"Integrate data between {source_adapter} and {target_adapter}"
        )

        decision = analysis.get("decision", {})
        confidence = analysis.get("confidence", 0.0)

        logger.info(f"LLM Decision: {decision} with confidence {confidence}")

        if confidence < 0.5:
            logger.warning("Low confidence in LLM decision. Aborting integration.")
            return {
                "analysis": analysis,
                "result": None,
                "status": "aborted_due_to_low_confidence",
                "timestamp": datetime.utcnow().isoformat()
            }

        action = decision.get("action_to_take", "default")
        required_transformations = decision.get("required_transformations", [])
        error_handling = decision.get("error_handling", {})
        success_criteria = decision.get("success_criteria", {})

        logger.info(f"Action to take: {action}")
        logger.info(f"Required transformations: {required_transformations}")
        logger.info(f"Error handling considerations: {error_handling}")
        logger.info(f"Success criteria: {success_criteria}")

        # Execute the integration based on the action
        if action == "default":
            logger.info("Performing default integration steps.")
        else:
            logger.info(f"Performing action: {action}")
            # Implement additional actions as needed

        # Execute the integration
        async with self.adapters[source_adapter] as source, \
                   self.adapters[target_adapter] as target:
            
            # Get data from source
            source_data = await source.execute()
            logger.info(f"Data from source ({source_adapter}): {source_data}")
            
            # Apply required transformations
            for transformation in required_transformations:
                # Implement transformation logic here
                logger.info(f"Applying transformation: {transformation}")
                # Example: if transformation == "map_fields", perform mapping

            # Transform and send to target
            result = await target.execute(source_data)
            logger.info(f"Result from target ({target_adapter}): {result}")

            # Evaluate success criteria
            # Implement success criteria evaluation here
            logger.info(f"Evaluating success criteria: {success_criteria}")

            return {
                "analysis": analysis,
                "result": result,
                "status": "success",
                "timestamp": datetime.utcnow().isoformat()
            }

    @classmethod
    def from_config(cls, config_path: str) -> 'LiquidAPI':
        """Create LiquidAPI instance from configuration file"""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        brain = LLMBrain(
            model_name=config.get('model_name', "HuggingFaceTB/SmolLM2-135M")
        )
        instance = cls(brain)
        
        # Load adapters from config
        for adapter_config in config.get('adapters', []):
            adapter_name = adapter_config['name']
            adapter_type = adapter_config.get('type', 'generic_adapter')
            adapter_class = cls.adapter_registry.get(adapter_type)
            if not adapter_class:
                logger.error(f"Adapter type '{adapter_type}' is not registered.")
                continue
            # Replace environment variable placeholders
            auth = adapter_config['config'].get('auth', {})
            for key, value in auth.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    auth[key] = os.getenv(env_var, "")
            adapter_config['config']['auth'] = auth
            adapter_instance = adapter_class(APIConfig(**adapter_config['config']))
            instance.register_adapter(adapter_name, adapter_instance)
        
        return instance

# Mock API server implementations

async def handle_crm(request):
    data = await request.json()
    logger.info(f"Mock CRM received data: {data}")
    return web.json_response({"status": "success", "message": "CRM data processed."})

async def handle_erp(request):
    data = await request.json()
    logger.info(f"Mock ERP received data: {data}")
    return web.json_response({"status": "success", "confirmation_number": "CONF123456"})

def run_mock_servers():
    app_crm = web.Application()
    app_crm.router.add_post('/data', handle_crm)
    runner_crm = web.AppRunner(app_crm)
    asyncio.create_task(start_server(runner_crm, 8081))

    app_erp = web.Application()
    app_erp.router.add_put('/update', handle_erp)
    runner_erp = web.AppRunner(app_erp)
    asyncio.create_task(start_server(runner_erp, 8082))

async def start_server(runner, port):
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', port)
    await site.start()
    logger.info(f"Mock server running on http://localhost:{port}")

# Example usage
async def main():
    # Start mock servers
    run_mock_servers()
    await asyncio.sleep(1)  # Wait a moment for servers to start

    # Example configuration (usually loaded from 'config.yml')
    example_config = {
        'model_name': "HuggingFaceTB/SmolLM2-135M",
        'adapters': [
            {
                'name': 'crm_adapter',
                'type': 'crm_adapter',
                'config': {
                    'url': 'http://localhost:8081/data',
                    'method': 'POST',
                    'headers': {'Content-Type': 'application/json'},
                    'auth': {'login': 'user', 'password': 'pass'},
                    'retry_attempts': 3,
                    'timeout': 10
                }
            },
            {
                'name': 'erp_adapter',
                'type': 'erp_adapter',
                'config': {
                    'url': 'http://localhost:8082/update',
                    'method': 'PUT',
                    'headers': {'Content-Type': 'application/json'},
                    'auth': {'login': 'erp_user', 'password': 'erp_pass'},
                    'retry_attempts': 3,
                    'timeout': 15
                }
            }
        ]
    }

    # For demonstration, write the example config to 'config.yml'
    with open('config.yml', 'w') as f:
        yaml.dump(example_config, f)

    # Create Liquid API instance
    liquid_api = LiquidAPI.from_config('config.yml')
    
    # Define integration context
    context = {
        "source_system": "CRM",
        "target_system": "ERP",
        "data_mapping": {
            "customer_id": "erp_customer_number",
            "order_details": "sales_order"
        }
    }
    
    # Execute integration
    try:
        result = await liquid_api.integrate(
            source_adapter="crm_adapter",
            target_adapter="erp_adapter",
            context=context
        )
        logger.info(f"Integration completed: {json.dumps(result, indent=2)}")
    except Exception as e:
        logger.error(f"Integration failed: {e}")

    # Keep the mock servers running for demonstration
    await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
