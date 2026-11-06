"""Launcher tests that stub flydsl, so they need neither a GPU nor flydsl."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import Mock
from unittest.mock import patch

from helion._compiler.flydsl import launcher


class TestFlydslLauncher(unittest.TestCase):
    def test_compile_only_does_not_launch_or_cache(self) -> None:
        # flyc.compile returns None under FlyDSL's COMPILE_ONLY mode.
        fx = types.ModuleType("flydsl.expr")
        fx.Stream = Mock(return_value="stream")  # pyrefly: ignore[missing-attribute]
        fx.Tensor = object  # pyrefly: ignore[missing-attribute]
        flyc = types.ModuleType("flydsl.compiler")
        flyc.jit = lambda fn: fn  # pyrefly: ignore[missing-attribute]
        flyc.compile = Mock(return_value=None)  # pyrefly: ignore[missing-attribute]
        flydsl = types.ModuleType("flydsl")
        flydsl.expr = fx  # pyrefly: ignore[missing-attribute]
        flydsl.compiler = flyc  # pyrefly: ignore[missing-attribute]
        modules = {"flydsl": flydsl, "flydsl.expr": fx, "flydsl.compiler": flyc}
        kernel = Mock()
        with (
            patch.dict(sys.modules, modules),
            patch.object(launcher, "_flydsl_jit_cache", {}),
            patch.object(launcher, "_flydsl_compiled_cache", {}) as compiled_cache,
            patch.object(launcher, "_flydsl_stream", None),
        ):
            for _ in range(2):
                self.assertIsNone(launcher.default_flydsl_launcher(kernel, (1,), 1, 2))
            self.assertEqual(compiled_cache, {})
        self.assertEqual(flyc.compile.call_count, 2)
        kernel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
