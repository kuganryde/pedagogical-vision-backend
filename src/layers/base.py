from abc import ABC, abstractmethod
from typing import Any

class BaseLayer(ABC):
    """Abstract base class for all modular vision & behavioral detection layers."""

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Process inputs and return structured pedagogical layer output."""
        pass
