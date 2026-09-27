"""Typed failures exposed by compression APIs and the command line."""


class CompressionError(RuntimeError):
    """Base class for an expected compression failure."""


class UnsupportedMethodError(CompressionError):
    """The requested representation cannot be produced safely."""


class IncompatibleArtifactError(CompressionError):
    """An input artifact does not satisfy a method's contract."""


class InvalidRecipeError(CompressionError):
    """A recipe is structurally valid YAML but scientifically invalid."""
