import pytest

from agentmerge.backends.algebra import (
    ExactDeltaFusionAlgebra,
    FactorComposition,
    FactorLinearFusionAlgebra,
    FusionModuleSpec,
    SourceAdapterDescriptor,
    create_fusion_algebra,
)
from agentmerge.core.policies import (
    GlobalGranularity,
    LayerGranularity,
    ModuleTarget,
    ScopedAdapterFusionPolicy,
)


def test_exact_delta_compiles_union_rank_and_layer_coefficients():
    policy = ScopedAdapterFusionPolicy(
        adapter_names=("a", "b"),
        granularity=LayerGranularity(2),
        scope_weights=(
            ("layer:0", (1.0, 0.0)),
            ("layer:1", (0.25, 0.75)),
        ),
    )
    sources = (
        SourceAdapterDescriptor("a", rank=2, native_scale=2.0),
        SourceAdapterDescriptor("b", rank=3, native_scale=0.5),
    )
    modules = (
        FusionModuleSpec("m0", ModuleTarget("layers.0.q_proj", 0, "q_proj")),
        FusionModuleSpec("m1", ModuleTarget("layers.1.q_proj", 1, "q_proj")),
    )

    plan = ExactDeltaFusionAlgebra().compile(policy, sources, modules)

    assert plan.algebra == "exact_delta"
    assert plan.factor_composition is FactorComposition.CONCATENATE
    assert plan.total_rank == 5
    assert [(item.source_name, item.start, item.end) for item in plan.slices] == [
        ("a", 0, 2),
        ("b", 2, 5),
    ]
    assert plan.modules[0].a_coefficients == (1.0, 1.0)
    assert plan.modules[0].b_coefficients == (2.0, 0.0)
    assert plan.modules[1].a_coefficients == (1.0, 1.0)
    assert plan.modules[1].b_coefficients == pytest.approx((0.5, 0.375))


def test_factor_linear_compiles_shared_rank_and_symmetric_coefficients():
    policy = ScopedAdapterFusionPolicy(
        adapter_names=("a", "b"),
        granularity=LayerGranularity(2),
        scope_weights=(
            ("layer:0", (1.0, 0.0)),
            ("layer:1", (0.25, 0.75)),
        ),
    )
    sources = (
        SourceAdapterDescriptor("a", rank=2, native_scale=2.0),
        SourceAdapterDescriptor("b", rank=3, native_scale=0.5),
    )
    modules = (
        FusionModuleSpec("m0", ModuleTarget("layers.0.q_proj", 0, "q_proj")),
        FusionModuleSpec("m1", ModuleTarget("layers.1.q_proj", 1, "q_proj")),
    )

    plan = FactorLinearFusionAlgebra().compile(policy, sources, modules)

    assert plan.algebra == "factor_linear"
    assert plan.factor_composition is FactorComposition.LINEAR_SUM
    assert plan.total_rank == 3
    assert [(item.source_name, item.start, item.end) for item in plan.slices] == [
        ("a", 0, 2),
        ("b", 0, 3),
    ]
    assert plan.modules[0].a_coefficients == pytest.approx((2.0**0.5, 0.0))
    assert plan.modules[0].b_coefficients == pytest.approx((2.0**0.5, 0.0))
    expected = (0.25 * 2.0**0.5, 0.75 * 0.5**0.5)
    assert plan.modules[1].a_coefficients == pytest.approx(expected)
    assert plan.modules[1].b_coefficients == pytest.approx(expected)


def test_exact_delta_is_the_default_registered_algebra():
    assert isinstance(create_fusion_algebra(), ExactDeltaFusionAlgebra)
    assert create_fusion_algebra("exact-delta").name == "exact_delta"
    assert isinstance(create_fusion_algebra("factor-linear"), FactorLinearFusionAlgebra)


def test_exact_delta_estimates_zero_rank_for_base_model_policy():
    policy = ScopedAdapterFusionPolicy(
        adapter_names=("a", "b"),
        granularity=GlobalGranularity(),
        scope_weights=(("global", (0.0, 0.0)),),
    )
    sources = (
        SourceAdapterDescriptor("a", rank=2, native_scale=1.0),
        SourceAdapterDescriptor("b", rank=3, native_scale=1.0),
    )

    assert ExactDeltaFusionAlgebra().estimate_rank(policy, sources) == 0
