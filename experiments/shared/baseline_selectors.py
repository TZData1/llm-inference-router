import random
import abc
import logging

logger = logging.getLogger(__name__)

class BaselineSelector(abc.ABC):
    """Abstract base class for baseline model selectors."""
    @abc.abstractmethod
    def select_model(self, query_id=None, context=None) -> str:
        """Selects a model ID based on the baseline strategy."""
        pass

class FixedModelSelector(BaselineSelector):
    """Selects a predetermined fixed model."""
    def __init__(self, model_id: str):
        if not model_id:
            raise ValueError("FixedModelSelector requires a valid model_id.")
        self.model_id = model_id
        logger.info(f"{self.__class__.__name__} initialized to always select model: {self.model_id}")

    def select_model(self, query_id=None, context=None) -> str:
        return self.model_id

class LargestModelSelector(FixedModelSelector):
    """Selects the pre-configured largest model."""

    pass

class SmallestModelSelector(FixedModelSelector):
    """Selects the pre-configured smallest model."""

    pass

class AccuracyOptimizedSelector(FixedModelSelector):
    """Selects the pre-configured best model for accuracy."""

    pass

class RandomModelSelector(BaselineSelector):
    """Randomly selects a model from the available pool."""
    def __init__(self, model_ids: list, seed: int = None):
        if not model_ids:
            raise ValueError("RandomModelSelector requires a non-empty list of model_ids.")
        self.model_ids = model_ids
        self.rng = random.Random(seed)
        logger.info(f"RandomModelSelector initialized with {len(self.model_ids)} models and seed={seed}")

    def select_model(self, query_id=None, context=None) -> str:
        return self.rng.choice(self.model_ids) 