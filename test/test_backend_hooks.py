from __future__ import annotations

import ast
from typing import ClassVar

import torch

import helion
from helion import backend_api
from helion._compiler import backend_registry
from helion._compiler.backend_registry import _REGISTRY
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
import helion.language as hl

_SCALAR_MARKER = "_scalar_marker"
_PROLOGUE_MARKER = "_prologue_marker"


class _PrologueBackend(backend_api.TritonBackend):
    # Reuses Triton codegen under a new name, like TileIRBackend.
    seen: ClassVar[list[tuple[int, int]]] = []

    @property
    def name(self) -> str:
        return "_test_prologue"

    @property
    def codegen_name(self) -> str:
        return "triton"

    def scalar_arg_preamble(self, arg: backend_api.Argument) -> list[ast.AST]:
        if arg.name == "alpha":
            return [backend_api.statement_from_string(f"{_SCALAR_MARKER} = 0")]
        return []

    def kernel_prologue(self, fn: backend_api.DeviceFunction) -> list[ast.stmt]:
        type(self).seen.append((len(fn.preamble), len(fn.body)))
        return [backend_api.statement_from_string(f"{_PROLOGUE_MARKER} = 0")]


class _FlagBackend(backend_api.TritonBackend):
    # Mirrors FlyDSL: body codegen sets a flag that kernel_prologue reads.
    calls: ClassVar[list[tuple[str, int]]] = []

    def __init__(self) -> None:
        super().__init__()
        self._needs_marker = False
        self._marker_emitted = False

    @property
    def name(self) -> str:
        return "_test_prologue_flag"

    @property
    def codegen_name(self) -> str:
        return "triton"

    def pre_codegen(
        self,
        graphs: list[backend_api.GraphInfo],
        config: backend_api.Config,
        tile_strategy: backend_api.TileStrategyDispatch,
    ) -> None:
        type(self).calls.append(("pre_codegen", id(self)))
        self._needs_marker = False
        self._marker_emitted = False
        super().pre_codegen(graphs, config, tile_strategy)

    def program_id_expr(self, dim: int, *, index_dtype: str) -> str:
        type(self).calls.append(("program_id_expr", id(self)))
        self._needs_marker = True
        return super().program_id_expr(dim, index_dtype=index_dtype)

    def kernel_prologue(self, fn: backend_api.DeviceFunction) -> list[ast.stmt]:
        type(self).calls.append(("kernel_prologue", id(self)))
        # Emit-once guard: only correct because pre_codegen resets it.
        if not self._needs_marker or self._marker_emitted:
            return []
        self._marker_emitted = True
        return [backend_api.statement_from_string(f"{_PROLOGUE_MARKER} = 0")]


class TestKernelPrologueDefault(TestCase):
    def test_default_is_empty(self) -> None:
        self.assertEqual(backend_api.TritonBackend().kernel_prologue(None), [])  # pyrefly: ignore[bad-argument-type]


@onlyBackends(["triton"])
class TestKernelPrologue(TestCase):
    def test_placement_and_call_order(self) -> None:
        _PrologueBackend.seen = []
        backend_api.register_compiler_backend(_PrologueBackend)
        try:

            @helion.kernel(backend="_test_prologue")
            def axpy(x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
                out = torch.empty_like(x)
                for tile in hl.tile(out.size()):
                    out[tile] = x[tile] + alpha * y[tile]
                return out

            x = torch.randn(64, 64, device=DEVICE)
            bound = axpy.bind((x, x, 2.0))
            config = bound.config_spec.default_config()
            # persistent_blocked puts the pid setup into fn.preamble.
            config.config["pid_type"] = "persistent_blocked"
            code = bound.to_triton_code(config)
        finally:
            _REGISTRY.pop("_test_prologue", None)

        # Called once, after both the preamble and the body were generated.
        self.assertEqual(len(_PrologueBackend.seen), 1)
        preamble_len, body_len = _PrologueBackend.seen[0]
        self.assertGreater(preamble_len, 0)
        self.assertGreater(body_len, 0)

        self.assertEqual(code.count(_PROLOGUE_MARKER), 1, code)
        (kernel,) = [
            node
            for node in ast.parse(code).body
            if isinstance(node, ast.FunctionDef)
            and any(_PROLOGUE_MARKER in ast.unparse(s) for s in node.body)
        ]
        stmts = [ast.unparse(s) for s in kernel.body]
        # scalar preamble -> kernel_prologue -> fn.preamble (pid setup) -> body
        self.assertEqual(stmts[0], f"{_SCALAR_MARKER} = 0", code)
        self.assertEqual(stmts[1], f"{_PROLOGUE_MARKER} = 0", code)
        self.assertTrue(stmts[2].startswith("total_pids"), code)

    def test_recompile_reemits_prologue(self) -> None:
        # One backend instance serves every pass of a bound kernel, so a flag
        # kept on the instance must be reset in pre_codegen (FlyDSL pattern).
        _FlagBackend.calls = []
        repaired_before = backend_registry._REPAIRED_CODEGEN_NAMES
        backend_api.register_compiler_backend(_FlagBackend)
        try:

            @helion.kernel(backend="_test_prologue_flag")
            def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                for tile in hl.tile(out.size()):
                    out[tile] = x[tile] + y[tile]
                return out

            x = torch.randn(64, 64, device=DEVICE)
            bound = add.bind((x, x))
            config = bound.config_spec.default_config()
            code1 = bound.to_triton_code(config)
            code2 = bound.to_triton_code(config)
        finally:
            _REGISTRY.pop("_test_prologue_flag", None)
            backend_registry._REPAIRED_CODEGEN_NAMES = repaired_before

        self.assertEqual(code1.count(_PROLOGUE_MARKER), 1, code1)
        self.assertEqual(code2.count(_PROLOGUE_MARKER), 1, code2)
        self.assertEqual(len({i for _, i in _FlagBackend.calls}), 1)
        hooks = [h for h, _ in _FlagBackend.calls]
        self.assertEqual(hooks.count("pre_codegen"), 2)
        self.assertEqual(hooks.count("kernel_prologue"), 2)


class _ThreadLevelHooksBackend(backend_api.TritonBackend):
    # Overrides the hooks no in-tree backend overrides (out-of-tree
    # thread-level backends such as helion-flydsl do), so Helion's own tests
    # exercise their non-default paths.
    calls: ClassVar[dict[str, list[object]]] = {}

    @property
    def name(self) -> str:
        return "_test_thread_level_hooks"

    @property
    def codegen_name(self) -> str:
        return "triton"

    def max_reduction_threads(self) -> int | None:
        return 64

    def reduction_block_size_is_inlined_constexpr(self) -> bool:
        return True

    def looped_reduction_thread_count(self, **kwargs: object) -> int | None:
        type(self).calls.setdefault("looped", []).append(kwargs["requested"])
        return 32

    def wrap_reduction_accumulator(self, acc_full: str, **kwargs: object) -> str:
        type(self).calls.setdefault("wrap", []).append(kwargs["thread_count"])
        return f"_wrapped_acc({acc_full})"

    def register_reduction_loop_config_slots(
        self, env: backend_api.CompileEnvironment, block_id: int, size_hint: int
    ) -> None:
        type(self).calls.setdefault("slots", []).append(size_hint)

    def expands_broadcast_dims(self) -> bool:
        return False

    def cast_scalar_ast(self, x: ast.AST, target_dtype: torch.dtype) -> ast.AST:
        type(self).calls.setdefault("cast_scalar", []).append(target_dtype)
        return backend_api.expr_from_string("_scalar_cast({x})", x=x)


def _rowsum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], device=x.device, dtype=x.dtype)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m] = x[tile_m, :].sum(-1)
    return out


def _add_row(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        out[tile_m, tile_n] = x[tile_m, tile_n] + b[tile_n]
    return out


def _sub_mean(x: torch.Tensor, begin: int, end: int) -> torch.Tensor:
    out = torch.zeros_like(x)
    for tile in hl.tile(begin, end):
        v = x[tile]
        # The mean divides by the tile's extent, a scalar from an index expr.
        out[tile] = v - v.mean()
    return out


def _kernel_params(code: str) -> list[str]:
    (kernel,) = [
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    ]
    return [arg.arg for arg in kernel.args.args]


@onlyBackends(["triton"])
class TestThreadLevelBackendHooks(TestCase):
    def setUp(self) -> None:
        super().setUp()
        _ThreadLevelHooksBackend.calls = {}
        backend_api.register_compiler_backend(_ThreadLevelHooksBackend)
        self.addCleanup(_REGISTRY.pop, "_test_thread_level_hooks", None)

    def _code(self, fn: object, args: tuple[torch.Tensor, ...], backend: str) -> str:
        bound = helion.kernel(backend=backend)(fn).bind(args)
        config = bound.config_spec.default_config()
        if "reduction_loops" in config.config:
            config.config["reduction_loops"] = [64]
        return bound.to_triton_code(config)

    def test_looped_reduction_hooks(self) -> None:
        x = torch.randn(32, 512, device=DEVICE)
        code = self._code(_rowsum, (x,), "_test_thread_level_hooks")
        calls = _ThreadLevelHooksBackend.calls
        self.assertEqual(calls["slots"], [512])
        self.assertEqual(calls["looped"], [64])
        # The thread count the backend returned is the one codegen used.
        self.assertEqual(calls["wrap"], [32])
        self.assertIn("_wrapped_acc(", code)
        # Inlined: a module-level constant, not a kernel parameter.
        self.assertIn("_REDUCTION_BLOCK_1 = tl.constexpr(64)", code)
        self.assertNotIn("_REDUCTION_BLOCK_1", _kernel_params(code))
        self.assertIn(
            "_REDUCTION_BLOCK_1", _kernel_params(self._code(_rowsum, (x,), "triton"))
        )

    def test_expands_broadcast_dims(self) -> None:
        x = torch.randn(32, 64, device=DEVICE)
        args = (x, x[0].contiguous())
        self.assertIn("load_1[None, :]", self._code(_add_row, args, "triton"))
        plugin = self._code(_add_row, args, "_test_thread_level_hooks")
        self.assertNotIn("load_1[None, :]", plugin)

    def test_cast_scalar_ast(self) -> None:
        args = (torch.randn(40, device=DEVICE), 3, 14)
        self.assertNotIn("_scalar_cast(", self._code(_sub_mean, args, "triton"))
        plugin = self._code(_sub_mean, args, "_test_thread_level_hooks")
        self.assertEqual(_ThreadLevelHooksBackend.calls["cast_scalar"], [torch.float32])
        self.assertIn("_scalar_cast(", plugin)


if __name__ == "__main__":
    import unittest

    unittest.main()
