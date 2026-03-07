# -*- coding: utf-8 -*-
from pkg_resources import DistributionNotFound, get_distribution
from .sampler import OriginalGenerator, Sampler, GANGenerator, ForestDiffusionGenerator, LLMGenerator
from .llm_config import LLMAPIConfig
from .llm_api_client import LLMAPIClient

__all__ = [
    "OriginalGenerator",
    "Sampler",
    "GANGenerator",
    "ForestDiffusionGenerator",
    "LLMGenerator",
    "LLMAPIConfig",
    "LLMAPIClient",
]

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound
