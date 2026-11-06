from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import ClassVar
from unittest import mock

import torch

import helion
from helion import backend_api
from helion import exc
from helion._compiler import backend_registry
from helion._compiler.backend_registry import _REGISTRY
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion.autotuner.local_cache import LocalAutotuneCache
from helion.autotuner.local_cache import StrictLocalAutotuneCache
from helion.autotuner.remote_cache import RemoteAutotuneCache
from helion.autotuner.remote_cache import RemoteCacheBackend
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator


class _VersionedBackend(backend_api.TritonBackend):
    version: ClassVar[str] = "1.0"

    @property
    def name(self) -> str:
        return "_test_versioned"

    @property
    def codegen_name(self) -> str:
        return "triton"

    def autotune_cache_key(self) -> str:
        return f"pkg=={self.version}"


class _BadKeyBackend(_VersionedBackend):
    result: ClassVar[object] = None

    @property
    def name(self) -> str:
        return "_test_bad_key"

    def autotune_cache_key(self) -> str:
        return self.result  # pyrefly: ignore[bad-return]


class _InMemoryRemote(RemoteCacheBackend):
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def put(self, key: str, data: str) -> None:
        self.store[key] = data


@contextlib.contextmanager
def _registered(cls: type[backend_api.Backend]) -> Iterator[None]:
    repaired_before = backend_registry._REPAIRED_CODEGEN_NAMES
    backend_api.register_compiler_backend(cls)
    try:
        yield
    finally:
        _REGISTRY.pop(cls().name, None)
        backend_registry._REPAIRED_CODEGEN_NAMES = repaired_before


def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


def _cache(
    backend: str, cls: type[LocalAutotuneCache] = LocalAutotuneCache
) -> tuple[LocalAutotuneCache, helion.Config]:
    x = torch.randn(64, 64, device=DEVICE)
    bound = helion.kernel(backend=backend)(_add).bind((x, x))
    cache = object.__new__(cls)
    cache.kernel = bound  # pyrefly: ignore[bad-assignment]
    cache.args = (x, x)  # pyrefly: ignore[bad-assignment]
    cache.autotuner = SimpleNamespace(settings=bound.settings)  # pyrefly: ignore[bad-assignment]
    cache.key = cache._generate_key()  # pyrefly: ignore[bad-assignment]
    return cache, bound.config_spec.default_config()


@onlyBackends(["triton"])
class TestBackendAutotuneCacheKey(TestCase):
    def test_default_leaves_key_unchanged(self) -> None:
        self.assertEqual(backend_api.TritonBackend().autotune_cache_key(), "")
        cache, _ = _cache("triton")
        self.assertEqual(cache.key.backend, "triton")
        self.assertEqual(cache.key.backend_autotune_key, "")
        self.assertNotIn("backend_autotune_key", repr(cache.key))

    def test_builtin_put_omits_empty_key_field(self) -> None:
        with (
            tempfile.TemporaryDirectory() as cache_dir,
            mock.patch.dict(os.environ, {"HELION_CACHE_DIR": cache_dir}),
        ):
            cache, config = _cache("triton")
            cache.put(config)
            data = json.loads(cache._get_local_cache_path().read_text())
        self.assertNotIn("backend_autotune_key", data["key"]["fields"])

    def test_non_str_key_raises(self) -> None:
        with _registered(_BadKeyBackend):
            for result in (None, 1):
                with self.subTest(result=result):
                    _BadKeyBackend.result = result
                    with self.assertRaisesRegex(TypeError, "_BadKeyBackend"):
                        _cache("_test_bad_key")

    def test_upgrade_misses_remote_cache(self) -> None:
        remote = _InMemoryRemote()
        with (
            _registered(_VersionedBackend),
            tempfile.TemporaryDirectory() as cache_dir,
            mock.patch.dict(os.environ, {"HELION_CACHE_DIR": cache_dir}),
        ):
            try:
                v1, config = _cache("_test_versioned", RemoteAutotuneCache)
                v1._backend = remote  # pyrefly: ignore[missing-attribute]
                v1.put(config)
                os.remove(v1._get_local_cache_path())  # leave only the remote copy
                self.assertIsNotNone(v1.get())

                _VersionedBackend.version = "2.0"
                v2, _ = _cache("_test_versioned", RemoteAutotuneCache)
                v2._backend = remote  # pyrefly: ignore[missing-attribute]
                self.assertIsNone(v2.get())
            finally:
                _VersionedBackend.version = "1.0"

    def test_upgrade_misses_cache(self) -> None:
        with (
            _registered(_VersionedBackend),
            tempfile.TemporaryDirectory() as cache_dir,
            mock.patch.dict(os.environ, {"HELION_CACHE_DIR": cache_dir}),
        ):
            try:
                for cls in (LocalAutotuneCache, StrictLocalAutotuneCache):
                    with self.subTest(cache=cls.__name__):
                        _VersionedBackend.version = "1.0"
                        v1, config = _cache("_test_versioned", cls)
                        self.assertEqual(v1.key.backend, "_test_versioned")
                        self.assertEqual(v1.key.backend_autotune_key, "pkg==1.0")
                        v1.put(config)
                        self.assertIsNotNone(v1.get())

                        _VersionedBackend.version = "2.0"
                        v2, _ = _cache("_test_versioned", cls)
                        self.assertIsNone(v2.get())

                        _VersionedBackend.version = "1.0"
                        self.assertIsNotNone(_cache("_test_versioned", cls)[0].get())
            finally:
                _VersionedBackend.version = "1.0"

    def test_assert_cache_hit_diff_defaults_only_omitted_fields(self) -> None:
        # A cache file lacking backend_autotune_key/search_policy_hash (pre-key
        # or built-in) matches "", but any other missing field is a difference.
        with (
            tempfile.TemporaryDirectory() as cache_dir,
            mock.patch.dict(
                os.environ,
                {"HELION_CACHE_DIR": cache_dir, "HELION_ASSERT_CACHE_HIT": "1"},
            ),
        ):
            cache, config = _cache("triton")
            self.assertEqual(cache.key.extra_cache_key, "")
            cache.put(config)
            path = cache._get_local_cache_path()
            data = json.loads(path.read_text())
            fields = data["key"]["fields"]
            self.assertNotIn("backend_autotune_key", fields)
            self.assertNotIn("search_policy_hash", fields)
            del fields["extra_cache_key"]
            path.with_stem("old").write_text(json.dumps(data))
            os.remove(path)
            stderr = io.StringIO()
            with (
                contextlib.redirect_stderr(stderr),
                self.assertRaises(exc.CacheAssertionError),
            ):
                cache.autotune()
        diff = stderr.getvalue().split("Key differences:")[1]
        self.assertIn("extra_cache_key:", diff)
        self.assertIn("<missing>", diff)
        self.assertNotIn("backend_autotune_key", diff)
        self.assertNotIn("search_policy_hash", diff)


if __name__ == "__main__":
    import unittest

    unittest.main()
