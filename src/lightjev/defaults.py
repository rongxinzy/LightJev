"""The standard, reproducible LightJev backbone."""
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def resolve_revision(model_name, revision=None):
    """Pin our standard model; never apply its commit to another model or path."""
    if revision is not None:
        return revision
    return DEFAULT_REVISION if model_name == DEFAULT_MODEL else None
