from lightjev.defaults import DEFAULT_MODEL, DEFAULT_REVISION, resolve_revision


def test_standard_backbone_is_pinned_without_affecting_other_models():
    assert DEFAULT_MODEL == 'Qwen/Qwen3-0.6B'
    assert len(DEFAULT_REVISION) == 40
    assert resolve_revision(DEFAULT_MODEL) == DEFAULT_REVISION
    assert resolve_revision(DEFAULT_MODEL, 'explicit-ref') == 'explicit-ref'
    assert resolve_revision('another/model') is None
    assert resolve_revision('/local/checkpoint') is None
