"""Native YAML/CLI policy validation, serialization and unsupported combinations."""

import json

import pytest
from pydantic import ValidationError

from speculators.train.config import TrainConfig
from speculators.train.config.schema import AnchorSamplingConfig


def _config(strategy="uniform", **policy):
    return TrainConfig(
        speculator_type="dflash",
        data={"anchor_sampling": {"strategy": strategy, **policy}},
    )


def test_defaults_remain_uniform_and_flatten_as_dict():
    cfg = TrainConfig()
    policy = cfg.flatten()["anchor_sampling"]
    assert policy == AnchorSamplingConfig().model_dump()
    assert policy["strategy"] == "uniform"
    assert TrainConfig.from_flat(cfg.flatten()).flatten() == cfg.flatten()


@pytest.mark.parametrize("strategy", [
    "target_flatness", "target_entropy_high", "target_entropy_low",
    "entropy_curriculum",
])
def test_static_policies_support_both_dflash_models(strategy):
    for model_type in ("dflash", "dflash2"):
        cfg = TrainConfig(speculator_type=model_type,
                          data={"anchor_sampling": {"strategy": strategy}})
        assert cfg.data.anchor_sampling.strategy == strategy


def test_native_yaml_cli_precedence_and_round_trip(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text("speculator_type: dflash\nverifier:\n  verifier_name_or_path: m\n"
                    "data:\n  anchor_sampling:\n    strategy: target_flatness\n"
                    "    retain_fraction: 0.5\n")
    yaml_cfg = TrainConfig.resolve(["--config", str(path)])
    assert yaml_cfg.flatten()["anchor_sampling"]["retain_fraction"] == 0.5
    cli_cfg = TrainConfig.resolve([
        "--config", str(path), "--anchor-sampling",
        json.dumps({"strategy": "target_entropy_low", "retain_fraction": 0.25}),
    ])
    assert cli_cfg.data.anchor_sampling.strategy == "target_entropy_low"
    assert cli_cfg.data.anchor_sampling.retain_fraction == 0.25
    assert cli_cfg.provenance["anchor_sampling"] == "flag"
    path.write_text(cli_cfg.dump_yaml())
    assert TrainConfig.resolve(["--config", str(path)]).flatten() == cli_cfg.flatten()


@pytest.mark.parametrize("value", ["[]", '"uniform"', "{bad json}"])
def test_cli_requires_json_mapping(value):
    with pytest.raises(SystemExit) as exc:
        TrainConfig.resolve(["--verifier-name-or-path", "m", "--anchor-sampling", value])
    assert exc.value.code == 2


@pytest.mark.parametrize("policy", [
    {"strategy": "imaginary"}, {"retain_fraction": 0},
    {"retain_fraction": 1.1}, {"retain_fraction": float("nan")},
    {"score_chunk_size": 0}, {"seed": -1}, {"typo": True},
    {"retain_fraction": 0.5}, {"score_backend": "bad"},
    {"curriculum_preset": "github"},
    {"strategy": "entropy_curriculum", "retain_fraction": 0.5},
    {"strategy": "target_flatness", "proposal": {"mode": "greedy"}},
    {"strategy": "round_occupancy"},
])
def test_policy_rejects_invalid_or_inapplicable_fields(policy):
    with pytest.raises(ValidationError):
        AnchorSamplingConfig(**policy)


@pytest.mark.parametrize("distribution", [
    {"mode": "softmax"},
    {"mode": "softmax", "temperature": 1.0, "top_k": 0},
    {"mode": "greedy", "temperature": 1.0},
    {"mode": "softmax", "temperature": 0, "top_k": 0, "top_p": 1.0},
    {"mode": "softmax", "temperature": 1, "top_k": -1, "top_p": 1.0},
    {"mode": "softmax", "temperature": 1, "top_k": 0, "top_p": 0},
    {"mode": "greedy", "unknown": True},
])
def test_occupancy_requires_explicit_valid_distributions(distribution):
    with pytest.raises(ValidationError):
        _config("round_occupancy", proposal=distribution, target={"mode": "greedy"})


def test_occupancy_accepts_greedy_and_explicit_softmax():
    cfg = _config("round_occupancy", proposal={"mode": "greedy"}, target={
        "mode": "softmax", "temperature": 0.6, "top_k": 0, "top_p": 0.95,
    })
    assert cfg.flatten()["anchor_sampling"]["target"]["top_p"] == 0.95


@pytest.mark.parametrize("model_type", ["eagle3", "dspark", "peagle", "mtp"])
def test_nonuniform_rejects_unsupported_architectures(model_type):
    with pytest.raises(ValidationError, match="requires dflash or dflash2"):
        TrainConfig(speculator_type=model_type,
                    data={"anchor_sampling": {"strategy": "target_flatness"}})


def test_occupancy_rejects_dflash2():
    with pytest.raises(ValidationError, match="supports dflash only"):
        TrainConfig(speculator_type="dflash2", data={"anchor_sampling": {
            "strategy": "round_occupancy", "proposal": {"mode": "greedy"},
            "target": {"mode": "greedy"},
        }})


@pytest.mark.parametrize("mapping", ["d2t_path", "t2d_path"])
def test_nonuniform_rejects_explicit_vocab_mapping(mapping):
    with pytest.raises(ValidationError, match="full vocabulary"):
        TrainConfig(speculator_type="dflash", draft={mapping: "mapping.npy"},
                    data={"anchor_sampling": {"strategy": "target_flatness"}})


def test_paper_curriculum_rejects_fully_masked_initial_epoch():
    with pytest.raises(ValidationError, match="epochs < 35"):
        TrainConfig(speculator_type="dflash", trainer={"epochs": 35},
                    data={"anchor_sampling": {"strategy": "entropy_curriculum"}})
    cfg = TrainConfig(speculator_type="dflash", trainer={"epochs": 35},
                      data={"anchor_sampling": {"strategy": "entropy_curriculum",
                                                "curriculum_preset": "github"}})
    assert cfg.trainer.epochs == 35


def test_score_backend_and_shared_chunk_configuration():
    assert AnchorSamplingConfig().score_chunk_size == 1024
    for backend in ("auto", "eager", "compile", "triton"):
        config = AnchorSamplingConfig(strategy="target_flatness", score_backend=backend)
        assert config.score_backend == backend
    # Uniform bypasses scoring; a shared default YAML may retain its chunk size.
    assert AnchorSamplingConfig(strategy="uniform", score_chunk_size=256).score_chunk_size == 256
