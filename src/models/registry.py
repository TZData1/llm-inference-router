# src/models/registry.py
class ModelRegistry:
    """Registry of available models for inference"""
    
    def __init__(self, models_config):
        self.models_config = models_config
        self.loaded_models = {}
    
    def get_model_info(self, model_id):
        """Get information about a specific model"""
        if model_id not in self.models_config:
            raise ValueError(f"Model {model_id} not found in registry")
            
        return self.models_config[model_id]
    
    def get_available_models(self):
        """Get list of all available model IDs"""
        return list(self.models_config.keys())
    
    def get_model(self, model_id):
        """Get or load a model for inference"""

        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
        print(f"Loading model {model_id}")
        self.loaded_models[model_id] = {"id": model_id}
        
        return self.loaded_models[model_id]
    
    def unload_model(self, model_id):
        """Unload a model to free resources"""
        if model_id in self.loaded_models:
            print(f"Unloading model {model_id}")
            del self.loaded_models[model_id]
            return True
            
        return False