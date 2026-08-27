# torch_tvarant.compiler

Graph compiler that fuses GEMM epilogues and pointwise chains for inference on
`device="tvarant"`.

```python
import torch_tvarant
compiled = torch_tvarant.compiler.compile(model)
y = compiled(x)
```

| API | Summary |
|---|---|
| [`compile`](compile.md) | Compile a module for inference |
| [`compile_fx`](compile_fx.md) | Fuse an FX graph |
| [`trace_module`](trace_module.md) | FX-trace with inlined Linear/activations |
| [`fuse_gemm_epilogue`](fuse_gemm_epilogue.md) | Fuse mm/linear + act into `linear_act` |
| [`fuse_pointwise`](fuse_pointwise.md) | Collapse pointwise chains |
| [`register`](register.md) | Register Dynamo backend `"tvarant"` |
| [`TvarantTracer`](TvarantTracer.md) | Custom FX tracer |
