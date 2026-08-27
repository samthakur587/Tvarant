"""Tvarant graph compiler: fuse GEMM epilogues and pointwise chains.

Two entry points:

* ``torch.compile(model, backend="tvarant")`` — Dynamo FX → fused kernels
* ``torch_tvarant.compiler.compile(module)`` — FX trace that inlines Linear/ReLU/SiLU

Fused kernels are JIT-compiled: the CPU simulator runs one fused loop; the
OpenCL backend emits and ``clBuildProgram``s a specialized kernel, then caches
it by program hash.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import torch.fx
import torch.nn as nn

# Must match csrc/jit/Jit.h PwOp.
LOAD, CONST, ADD, MUL, RELU, SILU, NEG, SCALE, SIGMOID = range(9)

last_log: dict[str, Any] = {}

_INLINE_MODULES = (
    nn.Sequential,
    nn.ModuleList,
    nn.ModuleDict,
    nn.Linear,
    nn.ReLU,
    nn.SiLU,
    nn.GELU,
    nn.LayerNorm,
    nn.Embedding,
    nn.Softmax,
    nn.Identity,
    nn.Dropout,
)


class TvarantTracer(torch.fx.Tracer):
    """Trace through Linear / activations so the fuser sees ATen ops."""

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, _INLINE_MODULES):
            return False
        return super().is_leaf_module(m, module_qualified_name)


def trace_module(module: nn.Module) -> torch.fx.GraphModule:
    """FX-trace ``module``, inlining Linear / common activations."""
    tracer = TvarantTracer()
    graph = tracer.trace(module)
    return torch.fx.GraphModule(module, graph)


def _norm(node: torch.fx.Node) -> str:
    if node.op == "call_method":
        return str(node.target)
    if node.op != "call_function":
        return ""
    return str(node.target).replace("aten::", "aten.")


def _aten_op(node: torch.fx.Node) -> str:
    """Best-effort ATen / builtin op name for an FX call_function node."""
    if node.op == "call_method":
        return str(node.target)
    if node.op != "call_function":
        return ""
    target = node.target
    # Prefer stable name attrs (torch 2.x builtins / OpOverloads).
    for attr in ("__name__", "name", "_name"):
        name = getattr(target, attr, None)
        if isinstance(name, str) and name:
            name = name.replace("aten::", "").replace("aten.", "")
            return name.split(".")[0]
    s = _norm(node)
    if not s:
        return ""
    s = s.replace("aten::", "").replace("aten.", "")
    # Fallback: "<built-in function linear>" / "<function relu at 0x...>"
    if "function " in s:
        s = s.split("function ", 1)[1]
        s = s.split(" at ", 1)[0].strip("<> ")
    return s.split(".")[0]


def _is_op(node: object, *keys: str) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    return _aten_op(node) in keys


def _act_name(node: torch.fx.Node) -> Optional[str]:
    name = _aten_op(node)
    if name == "silu":
        return "silu"
    if name in ("relu", "relu_"):
        return "relu"
    return None


def _pw_kind(node: torch.fx.Node) -> Optional[tuple[str, int]]:
    name = _aten_op(node)
    if name in ("silu",):
        return ("unary", SILU)
    if name in ("relu", "relu_"):
        return ("unary", RELU)
    if name == "neg":
        return ("unary", NEG)
    if name == "sigmoid":
        return ("unary", SIGMOID)
    if name == "add":
        return ("binary", ADD)
    if name == "mul":
        # mul.Scalar vs mul.Tensor — inspect args
        if node.args and not isinstance(node.args[-1] if len(node.args) > 1 else None, torch.fx.Node):
            if len(node.args) > 1 and not isinstance(node.args[1], torch.fx.Node):
                return ("scalar", SCALE)
        if "Scalar" in _norm(node):
            return ("scalar", SCALE)
        return ("binary", MUL)
    return None


def fuse_gemm_epilogue(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Fuse mm/addmm/linear + optional bias + relu/silu into linear_act."""
    graph = gm.graph
    fused = 0
    for node in list(graph.nodes):
        act = _act_name(node)
        if act is None or not node.args:
            continue
        prod = node.args[0]
        if not isinstance(prod, torch.fx.Node) or len(prod.users) != 1:
            continue

        def emit(x, w, bias, trans_b: bool) -> None:
            nonlocal fused
            with graph.inserting_before(node):
                out = graph.call_function(
                    torch.ops.tvarant.linear_act,
                    args=(x, w, bias, act, trans_b),
                )
            node.replace_all_uses_with(out)
            graph.erase_node(node)
            fused += 1

        if _is_op(prod, "linear"):
            x, w = prod.args[0], prod.args[1]
            bias = prod.args[2] if len(prod.args) > 2 else prod.kwargs.get("bias")
            emit(x, w, bias, True)
            graph.erase_node(prod)
            continue

        if _is_op(prod, "addmm"):
            alpha = prod.kwargs.get("alpha", 1)
            beta = prod.kwargs.get("beta", 1)
            try:
                if float(alpha) != 1.0 or float(beta) != 1.0:
                    continue
            except (TypeError, ValueError):
                continue
            bias, x, w = prod.args[0], prod.args[1], prod.args[2]
            emit(x, w, bias, False)
            graph.erase_node(prod)
            continue

        if _is_op(prod, "mm"):
            x, w = prod.args[0], prod.args[1]
            emit(x, w, None, False)
            graph.erase_node(prod)
            continue

        if _is_op(prod, "add") and len(prod.args) >= 2:
            left, right = prod.args[0], prod.args[1]
            gemm = bias = None
            for cand, other in ((left, right), (right, left)):
                if _is_op(cand, "mm") and len(cand.users) == 1:
                    gemm, bias = cand, other
                    break
            if gemm is None:
                continue
            x, w = gemm.args[0], gemm.args[1]
            emit(x, w, bias, False)
            graph.erase_node(prod)
            graph.erase_node(gemm)

    graph.lint()
    gm.recompile()
    last_log["gemm_epilogue"] = fused
    return gm


def fuse_pointwise(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Collapse connected pointwise subgraphs into one JIT kernel."""
    graph = gm.graph
    fused_groups = 0
    for root in list(graph.nodes):
        if _pw_kind(root) is None:
            continue
        if any(_pw_kind(u) is not None for u in root.users if isinstance(u, torch.fx.Node)):
            continue

        fused = {root}
        changed = True
        while changed:
            changed = False
            for n in graph.nodes:
                if n in fused or _pw_kind(n) is None:
                    continue
                users = [u for u in n.users if isinstance(u, torch.fx.Node)]
                if users and all(u in fused for u in users):
                    fused.add(n)
                    changed = True
        if len(fused) < 2:
            continue

        order = [n for n in graph.nodes if n in fused]
        ssa: dict[torch.fx.Node, int] = {}
        inputs: list[torch.fx.Node] = []
        ops: list[int] = []
        a: list[int] = []
        b: list[int] = []
        input_ids: list[int] = []
        alphas: list[float] = []
        consts: list[float] = []

        def emit_load(n: torch.fx.Node) -> int:
            if n in ssa:
                return ssa[n]
            ops.append(LOAD)
            a.append(-1)
            b.append(-1)
            input_ids.append(len(inputs))
            alphas.append(1.0)
            consts.append(0.0)
            inputs.append(n)
            ssa[n] = len(ops) - 1
            return ssa[n]

        def rec_input(src: object) -> int:
            if isinstance(src, torch.fx.Node):
                if src in fused:
                    return ssa[src]
                return emit_load(src)
            ops.append(CONST)
            a.append(-1)
            b.append(-1)
            input_ids.append(-1)
            alphas.append(1.0)
            consts.append(float(src))  # type: ignore[arg-type]
            return len(ops) - 1

        ok = True
        for n in order:
            kind_op = _pw_kind(n)
            if kind_op is None:
                ok = False
                break
            kind, op = kind_op
            if kind == "unary":
                ia = rec_input(n.args[0])
                ops.append(op)
                a.append(ia)
                b.append(-1)
                input_ids.append(-1)
                alphas.append(1.0)
                consts.append(0.0)
            elif kind == "binary":
                ia = rec_input(n.args[0])
                ib = rec_input(n.args[1])
                alpha = 1.0
                if len(n.args) > 2:
                    alpha = float(n.args[2])  # type: ignore[arg-type]
                alpha = float(n.kwargs.get("alpha", alpha))
                ops.append(op)
                a.append(ia)
                b.append(ib)
                input_ids.append(-1)
                alphas.append(alpha)
                consts.append(0.0)
            elif kind == "scalar":
                ia = rec_input(n.args[0])
                ops.append(SCALE)
                a.append(ia)
                b.append(-1)
                input_ids.append(-1)
                alphas.append(float(n.args[1]))  # type: ignore[arg-type]
                consts.append(0.0)
            else:
                ok = False
                break
            ssa[n] = len(ops) - 1
        if not ok or not inputs:
            continue

        with graph.inserting_before(root):
            new = graph.call_function(
                torch.ops.tvarant.pointwise,
                args=(inputs, ops, a, b, input_ids, alphas, consts),
            )
        root.replace_all_uses_with(new)
        for n in reversed(order):
            if not n.users:
                graph.erase_node(n)
        fused_groups += 1

    graph.lint()
    gm.recompile()
    last_log["pointwise_groups"] = fused_groups
    return gm


def compile_fx(gm: torch.fx.GraphModule, example_inputs: Any = None) -> torch.fx.GraphModule:
    """Optimize a captured FX graph for Tvarant."""
    _ = example_inputs
    gm = fuse_gemm_epilogue(gm)
    gm = fuse_pointwise(gm)
    return gm


def _dynamo_backend(gm: torch.fx.GraphModule, example_inputs: Any):
    return compile_fx(gm, example_inputs)


_registered = False


def register() -> None:
    """Register ``backend="tvarant"`` with torch.compile / Dynamo."""
    global _registered
    if _registered:
        return
    try:
        from torch._dynamo.backends.registry import register_backend

        register_backend(_dynamo_backend, name="tvarant")
        _registered = True
        return
    except Exception:
        pass
    try:
        import torch._dynamo as dynamo

        if hasattr(dynamo, "register_backend"):
            dynamo.register_backend(_dynamo_backend, name="tvarant")
            _registered = True
    except Exception:
        pass


def compile(module: torch.nn.Module, *, dynamic: bool = False) -> Callable:
    """Compile a module for Tvarant inference.

    Prefers FX tracing that inlines Linear / activations (stable on this custom
    device). Falls back to ``torch.compile(..., backend="tvarant")``.
    """
    module.eval()
    try:
        return compile_fx(trace_module(module))
    except Exception:
        return torch.compile(module, backend="tvarant", dynamic=dynamic)


register()
