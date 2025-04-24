from typing import Dict, Any, Type, Optional, Callable
import torch.nn as nn


class ModelRegistry:
    """Registry for model classes and factory functions."""

    _models: Dict[str, Type[nn.Module]] = {}
    _factory_funcs: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[nn.Module]) -> None:
        """
        Register a model class.

        Args:
            name: Name of the model
            model_class: Model class to register
        """
        cls._models[name.lower()] = model_class

    @classmethod
    def register_factory(cls, name: str, factory_func: Callable) -> None:
        """
        Register a factory function.

        Args:
            name: Name of the model
            factory_func: Factory function to create the model
        """
        cls._factory_funcs[name.lower()] = factory_func

    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> nn.Module:
        """
        Create a model instance.

        Args:
            name: Name of the model
            config: Configuration for the model

        Returns:
            Model instance

        Raises:
            ValueError: If model is not registered
        """
        name = name.lower()

        # First check for factory function
        if name in cls._factory_funcs:
            return cls._factory_funcs[name](config)

        # Then check for model class
        if name in cls._models:
            return cls._models[name](config)

        raise ValueError(f"Model {name} not registered")

    @classmethod
    def list_models(cls) -> Dict[str, Type[nn.Module]]:
        """
        List registered models.

        Returns:
            Dictionary of registered models
        """
        return cls._models.copy()

    @classmethod
    def list_factories(cls) -> Dict[str, Callable]:
        """
        List registered factory functions.

        Returns:
            Dictionary of registered factory functions
        """
        return cls._factory_funcs.copy()