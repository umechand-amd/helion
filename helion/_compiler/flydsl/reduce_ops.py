"""FlyDSL-backend codegen for ops defined in ``helion.language.reduce_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "flydsl")``
registrations; ``_codegen_modules`` imports it so registration keeps the same
eager timing as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ... import exc
from ...language import _decorators
from ...language.reduce_ops import _reduce

if TYPE_CHECKING:
    import ast

    from ..inductor_lowering import CodegenState


@_decorators.codegen(_reduce, "flydsl")
def _(state: CodegenState) -> ast.AST | list[ast.AST]:
    from ..ast_extension import expr_from_string
    from ..ast_extension import statement_from_string

    combine_graph_id = state.proxy_arg(0)
    dim = state.proxy_arg(2)
    assert isinstance(combine_graph_id, int)

    if dim is None:
        raise exc.BackendUnsupported("flydsl", "hl.reduce(..., dim=None)")

    from ..cute.reduce_ops import _infer_builtin_reduction_type_for_cute

    reduction_type = _infer_builtin_reduction_type_for_cute(state, combine_graph_id)
    if reduction_type is None:
        raise exc.BackendUnsupported("flydsl", "hl.reduce custom combine function")

    input_var = state.codegen.lift(state.ast_arg(1), dce=True, prefix="reduce_in").id

    _WARP = 64  # ROCm wavefront size
    _STEPS = 6  # log2(64)

    r = state.codegen.tmpvar(prefix="reduce_out")
    state.add_statement(statement_from_string(f"{r} = {input_var}"))

    for step in range(_STEPS):
        off = _WARP >> (step + 1)
        if reduction_type == "sum":
            # arith.FastMathFlags is the correct namespace; fmath.FastMathFlags
            # does not exist (flydsl.expr.math has no FastMathFlags attribute).
            state.add_statement(
                statement_from_string(
                    f"{r} = {r}.addf({r}.shuffle_xor({off}, {_WARP}), fastmath=arith.FastMathFlags.fast)"
                )
            )
        elif reduction_type == "max":
            state.add_statement(
                statement_from_string(
                    f"{r} = {r}.maximumf({r}.shuffle_xor({off}, {_WARP}))"
                )
            )
        elif reduction_type == "min":
            # flydsl Vector has no .minimumf(); use min(a,b) = -max(-a,-b).
            state.add_statement(
                statement_from_string(
                    f"{r} = -({r}.shuffle_xor({off}, {_WARP})).maximumf(-{r})"
                )
            )
        else:
            raise exc.BackendUnsupported("flydsl", f"reduction {reduction_type!r}")

    return expr_from_string(r)
