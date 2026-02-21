"""Core abstractions: domains, fields, DOFs, and base classes."""

from simulacrum.core.domain import Domain
from simulacrum.core.field import Field
from simulacrum.core.dof import DOFMap
from simulacrum.core.logging import get_logger

__all__ = ["Domain", "Field", "DOFMap", "get_logger"]
