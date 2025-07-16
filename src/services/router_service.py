# src/services/router_service.py
import importlib



class RouterService:
    """Service for selecting models using MAB algorithms"""

    ALGORITHM_MAP = {
        "epsilon_greedy": "EpsilonGreedy",
        "contextual_epsilon_greedy": "ContextualEpsilonGreedy",
        "linucb": "LinUCB",
        "thompson_sampling": "ThompsonSampling",
    }

    def __init__(
        self, algorithm_name, model_ids, algorithm_params=None, context_dimension=10
    ):
        self.algorithm_name = algorithm_name
        self.model_ids = model_ids
        self.algorithm_params = algorithm_params or {}
        self.bandit = self._create_bandit(
            algorithm_name, model_ids, context_dimension=context_dimension
        )

    def _create_bandit(self, algorithm_name, model_ids, context_dimension):
        """Create a bandit algorithm instance using explicit mapping"""
        try:
            module_name = algorithm_name.lower()
            class_name = self.ALGORITHM_MAP.get(algorithm_name.lower())
            if not class_name:
                class_name = "".join(
                    word.capitalize() for word in algorithm_name.split("_")
                )
            module = importlib.import_module(f"src.bandit.{module_name}")
            bandit_class = getattr(module, class_name)
            return bandit_class(
                model_ids, **self.algorithm_params, context_dimension=context_dimension
            )
        except (ImportError, AttributeError) as e:
            print(f"Error creating bandit algorithm {algorithm_name}: {e}")
            from src.bandit.epsilon_greedy import EpsilonGreedy

            return EpsilonGreedy(model_ids)

    def select_model(self, features):
        """Select model based on features"""
        context = features.get("context_vector") if features else None
        return self.bandit.select_model(context)

    def update(self, context, model_id, reward):
        """Update the bandit algorithm with the observed reward."""

        self.bandit.update(model_id, reward, context=context)
