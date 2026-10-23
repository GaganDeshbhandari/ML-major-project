"""Unit tests for src.factllama.model_loader.

All heavy dependencies are injected as fakes: no torch/transformers/peft
import, no network, no GPU, no weights.
"""

import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import src.factllama.model_loader as ml


def _write_adapter(tmp_path: Path, *, with_tokenizer: bool = False) -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / ml.ADAPTER_CONFIG_FILENAME).write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_text("", encoding="utf-8")
    if with_tokenizer:
        (adapter / ml.TOKENIZER_CONFIG_FILENAME).write_text(
            "{}", encoding="utf-8"
        )
    return adapter


def _fake_environment(
    *,
    cuda_available: bool = True,
    model_error: Exception | None = None,
    tokenizer_error: Exception | None = None,
    adapter_error: Exception | None = None,
):
    model = MagicMock(name="model")
    tokenizer = SimpleNamespace(
        pad_token=None, eos_token="</s>", pad_token_id=None, eos_token_id=2
    )

    auto_tokenizer = MagicMock(name="AutoTokenizer")
    auto_tokenizer.from_pretrained = MagicMock(return_value=tokenizer)
    if tokenizer_error is not None:
        auto_tokenizer.from_pretrained.side_effect = tokenizer_error

    auto_model = MagicMock(name="AutoModelForCausalLM")
    auto_model.from_pretrained = MagicMock(return_value=model)
    if model_error is not None:
        auto_model.from_pretrained.side_effect = model_error

    bnb_config = MagicMock(name="BitsAndBytesConfig", return_value="bnb-config")
    transformers = SimpleNamespace(
        AutoTokenizer=auto_tokenizer,
        AutoModelForCausalLM=auto_model,
        BitsAndBytesConfig=bnb_config,
    )

    peft_model = MagicMock(name="PeftModel")
    peft_model.from_pretrained = MagicMock(return_value=model)
    if adapter_error is not None:
        peft_model.from_pretrained.side_effect = adapter_error
    peft = SimpleNamespace(PeftModel=peft_model)

    torch_module = SimpleNamespace(
        bfloat16="BF16",
        float16="FP16",
        float32="FP32",
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
    )

    modules = {
        "torch": torch_module,
        "transformers": transformers,
        "peft": peft,
        "accelerate": object(),
        "bitsandbytes": object(),
    }
    handles = {
        "model": model,
        "tokenizer": tokenizer,
        "auto_tokenizer": auto_tokenizer,
        "auto_model": auto_model,
        "bnb_config": bnb_config,
        "peft_model": peft_model,
    }
    return modules, handles


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_defaults_when_config_file_is_empty(tmp_path):
    config_file = tmp_path / "factllama.json"
    config_file.write_text("{}", encoding="utf-8")

    config = ml.load_config(path=config_file, env={})

    assert config.base_model == ml.DEFAULT_BASE_MODEL
    assert config.adapter_path == Path(ml.DEFAULT_ADAPTER_PATH)
    assert config.load_in_4bit is True
    assert config.max_new_tokens == 16
    assert config.do_sample is False


def test_json_values_are_loaded(tmp_path):
    config_file = tmp_path / "factllama.json"
    config_file.write_text(
        """
        {
          "base_model": "local/base",
          "adapter_path": "some/adapter",
          "load_in_4bit": false,
          "max_new_tokens": 8,
          "temperature": 0.2
        }
        """,
        encoding="utf-8",
    )

    config = ml.load_config(path=config_file, env={})

    assert config.base_model == "local/base"
    assert config.adapter_path == Path("some/adapter")
    assert config.load_in_4bit is False
    assert config.max_new_tokens == 8
    assert config.temperature == 0.2


def test_environment_overrides_file(tmp_path):
    config_file = tmp_path / "factllama.json"
    config_file.write_text('{"base_model": "from/file"}', encoding="utf-8")

    config = ml.load_config(
        path=config_file,
        env={
            "FACTLLAMA_BASE_MODEL": "from/env",
            "FACTLLAMA_ADAPTER_PATH": "env/adapter",
            "FACTLLAMA_LOAD_IN_4BIT": "false",
            "FACTLLAMA_MAX_NEW_TOKENS": "32",
            "FACTLLAMA_TEMPERATURE": "0.5",
        },
    )

    assert config.base_model == "from/env"
    assert config.adapter_path == Path("env/adapter")
    assert config.load_in_4bit is False
    assert config.max_new_tokens == 32
    assert config.temperature == 0.5


def test_blank_environment_values_are_ignored(tmp_path):
    config_file = tmp_path / "factllama.json"
    config_file.write_text("{}", encoding="utf-8")

    config = ml.load_config(
        path=config_file, env={"FACTLLAMA_BASE_MODEL": "  "}
    )

    assert config.base_model == ml.DEFAULT_BASE_MODEL


def test_missing_explicit_config_is_an_error(tmp_path):
    with pytest.raises(ml.ConfigurationError):
        ml.load_config(path=tmp_path / "missing.json", env={})


def test_malformed_json_is_an_error(tmp_path):
    config_file = tmp_path / "factllama.json"
    config_file.write_text("{not json", encoding="utf-8")

    with pytest.raises(ml.ConfigurationError):
        ml.load_config(path=config_file, env={})


def test_non_object_json_is_an_error(tmp_path):
    config_file = tmp_path / "factllama.json"
    config_file.write_text("[]", encoding="utf-8")

    with pytest.raises(ml.ConfigurationError):
        ml.load_config(path=config_file, env={})


def test_unknown_key_is_an_error(tmp_path):
    config_file = tmp_path / "factllama.json"
    config_file.write_text('{"unknown_key": 1}', encoding="utf-8")

    with pytest.raises(ml.ConfigurationError) as excinfo:
        ml.load_config(path=config_file, env={})
    assert "unknown_key" in str(excinfo.value)


@pytest.mark.parametrize(
    "env",
    [
        {"FACTLLAMA_LOAD_IN_4BIT": "perhaps"},
        {"FACTLLAMA_MAX_NEW_TOKENS": "0"},
        {"FACTLLAMA_MAX_NEW_TOKENS": "many"},
        {"FACTLLAMA_TEMPERATURE": "-1"},
        {"FACTLLAMA_BNB_4BIT_QUANT_TYPE": "int4"},
    ],
)
def test_invalid_values_are_errors(tmp_path, env):
    config_file = tmp_path / "factllama.json"
    config_file.write_text("{}", encoding="utf-8")

    with pytest.raises(ml.ConfigurationError):
        ml.load_config(path=config_file, env=env)


def test_relative_adapter_path_resolves_against_project_root():
    resolved = ml.resolve_adapter_path("models/factllama/x")
    assert resolved == (ml.project_root() / "models/factllama/x").resolve()


def test_absolute_adapter_path_is_kept(tmp_path):
    assert ml.resolve_adapter_path(tmp_path) == tmp_path


def test_validate_adapter_path_missing_directory(tmp_path):
    with pytest.raises(ml.AdapterNotFoundError) as excinfo:
        ml.validate_adapter_path(tmp_path / "nope")
    assert "factllama_final_adapter_v2" in str(excinfo.value)


def test_validate_adapter_path_missing_weights(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / ml.ADAPTER_CONFIG_FILENAME).write_text("{}", encoding="utf-8")

    with pytest.raises(ml.AdapterNotFoundError) as excinfo:
        ml.validate_adapter_path(adapter)
    assert "adapter_model" in str(excinfo.value)


def test_validate_adapter_path_accepts_bin_weights(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / ml.ADAPTER_CONFIG_FILENAME).write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.bin").write_text("", encoding="utf-8")

    assert ml.validate_adapter_path(adapter) == adapter


# ---------------------------------------------------------------------------
# Lazy loading and dependency checks
# ---------------------------------------------------------------------------


def test_constructing_loader_does_not_import_dependencies(tmp_path):
    adapter = _write_adapter(tmp_path)
    config = ml.FactLLaMAConfig(adapter_path=adapter)

    with patch.object(ml, "_import_module") as import_module:
        loader = ml.FactLLaMAModelLoader(config)

    import_module.assert_not_called()
    assert loader.is_loaded is False


def test_package_import_does_not_import_torch():
    code = (
        "import sys; import src.factllama; "
        "assert 'torch' not in sys.modules, 'torch imported eagerly'; "
        "assert 'transformers' not in sys.modules, 'transformers imported eagerly'"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ml.project_root(),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_missing_dependency_raises_clear_error(tmp_path):
    adapter = _write_adapter(tmp_path)
    loader = ml.FactLLaMAModelLoader(ml.FactLLaMAConfig(adapter_path=adapter))

    with patch.object(
        ml,
        "_import_module",
        side_effect=ml.MissingDependencyError("missing peft"),
    ):
        with pytest.raises(ml.MissingDependencyError):
            loader.load()


def test_four_bit_requires_cuda(tmp_path):
    adapter = _write_adapter(tmp_path)
    modules, handles = _fake_environment(cuda_available=False)
    loader = ml.FactLLaMAModelLoader(
        ml.FactLLaMAConfig(adapter_path=adapter), modules=modules
    )

    with pytest.raises(ml.QuantizationUnsupportedError):
        loader.load()

    handles["auto_model"].from_pretrained.assert_not_called()


def test_four_bit_requires_bitsandbytes(tmp_path):
    adapter = _write_adapter(tmp_path)
    modules, _ = _fake_environment()
    modules.pop("bitsandbytes")
    loader = ml.FactLLaMAModelLoader(
        ml.FactLLaMAConfig(adapter_path=adapter), modules=modules
    )

    with pytest.raises(ml.MissingDependencyError):
        loader.load()


# ---------------------------------------------------------------------------
# Successful loading
# ---------------------------------------------------------------------------


def test_loads_base_model_and_attaches_adapter(tmp_path):
    adapter = _write_adapter(tmp_path)
    modules, handles = _fake_environment()
    loader = ml.FactLLaMAModelLoader(
        ml.FactLLaMAConfig(adapter_path=adapter), modules=modules
    )

    loaded = loader.load()

    assert loaded.model is handles["model"]
    assert loaded.tokenizer is handles["tokenizer"]
    assert loaded.adapter_path == adapter
    assert handles["tokenizer"].pad_token == "</s>"
    assert loader.is_loaded is True

    handles["bnb_config"].assert_called_once_with(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="BF16",
        bnb_4bit_use_double_quant=True,
    )
    handles["auto_model"].from_pretrained.assert_called_once_with(
        ml.DEFAULT_BASE_MODEL,
        device_map="auto",
        trust_remote_code=False,
        quantization_config="bnb-config",
    )
    handles["peft_model"].from_pretrained.assert_called_once_with(
        handles["model"], str(adapter)
    )
    handles["model"].eval.assert_called_once()


def test_tokenizer_loaded_from_adapter_when_present(tmp_path):
    adapter = _write_adapter(tmp_path, with_tokenizer=True)
    modules, handles = _fake_environment()
    loader = ml.FactLLaMAModelLoader(
        ml.FactLLaMAConfig(adapter_path=adapter), modules=modules
    )

    loader.load()

    handles["auto_tokenizer"].from_pretrained.assert_called_once_with(
        str(adapter)
    )


def test_second_load_is_cached(tmp_path):
    adapter = _write_adapter(tmp_path)
    modules, handles = _fake_environment()
    loader = ml.FactLLaMAModelLoader(
        ml.FactLLaMAConfig(adapter_path=adapter), modules=modules
    )

    first = loader.load()
    second = loader.load()

    assert first is second
    assert handles["auto_model"].from_pretrained.call_count == 1


def test_non_four_bit_mode_skips_quantization(tmp_path):
    adapter = _write_adapter(tmp_path)
    modules, handles = _fake_environment(cuda_available=False)
    modules.pop("bitsandbytes")
    loader = ml.FactLLaMAModelLoader(
        ml.FactLLaMAConfig(adapter_path=adapter, load_in_4bit=False),
        modules=modules,
    )

    loader.load()

    handles["bnb_config"].assert_not_called()
    kwargs = handles["auto_model"].from_pretrained.call_args.kwargs
    assert "quantization_config" not in kwargs


# ---------------------------------------------------------------------------
# Failure mapping
# ---------------------------------------------------------------------------


def test_gated_model_error_becomes_model_access_error(tmp_path):
    adapter = _write_adapter(tmp_path)
    modules, _ = _fake_environment(
        model_error=OSError(
            "Repository is gated. Please authenticate with a token."
        )
    )
    loader = ml.FactLLaMAModelLoader(
        ml.FactLLaMAConfig(adapter_path=adapter), modules=modules
    )

    with pytest.raises(ml.ModelAccessError) as excinfo:
        loader.load()
    assert "HF_TOKEN" in str(excinfo.value)


def test_other_model_errors_become_model_load_error(tmp_path):
    adapter = _write_adapter(tmp_path)
    modules, _ = _fake_environment(
        model_error=RuntimeError("unexpected tensor shape")
    )
    loader = ml.FactLLaMAModelLoader(
        ml.FactLLaMAConfig(adapter_path=adapter), modules=modules
    )

    with pytest.raises(ml.ModelLoadError):
        loader.load()


def test_tokenizer_access_error_is_mapped(tmp_path):
    adapter = _write_adapter(tmp_path)
    modules, _ = _fake_environment(
        tokenizer_error=ValueError("not a valid model identifier")
    )
    loader = ml.FactLLaMAModelLoader(
        ml.FactLLaMAConfig(adapter_path=adapter), modules=modules
    )

    with pytest.raises(ml.ModelAccessError):
        loader.load()


def test_adapter_attach_error_is_mapped(tmp_path):
    adapter = _write_adapter(tmp_path)
    modules, _ = _fake_environment(
        adapter_error=RuntimeError("adapter keys do not match")
    )
    loader = ml.FactLLaMAModelLoader(
        ml.FactLLaMAConfig(adapter_path=adapter), modules=modules
    )

    with pytest.raises(ml.ModelLoadError):
        loader.load()


def test_unknown_compute_dtype_is_a_config_error(tmp_path):
    adapter = _write_adapter(tmp_path)
    modules, _ = _fake_environment()
    config = ml.FactLLaMAConfig(
        adapter_path=adapter, bnb_4bit_compute_dtype="int3"
    )
    loader = ml.FactLLaMAModelLoader(config, modules=modules)

    with pytest.raises(ml.ConfigurationError):
        loader.load()


def test_missing_adapter_stops_before_imports(tmp_path):
    loader = ml.FactLLaMAModelLoader(
        ml.FactLLaMAConfig(adapter_path=tmp_path / "missing")
    )

    with patch.object(ml, "_import_module") as import_module:
        with pytest.raises(ml.AdapterNotFoundError):
            loader.load()

    import_module.assert_not_called()
