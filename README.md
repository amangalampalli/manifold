# manifold

Research prototype for graph signal stabilization with neural sheaf ODEs.

The default experiment uses synthetic community graphs, fixed-step RK4
integration, sparse sheaf consistency operators, active sensing, and a minimax
source/controller training loop. Apple Metal/MPS is selected automatically when
available, with CPU fallback paths for sparse operations.

```bash
conda run -n manifold manifold-train --config configs/default.yaml --device auto
```
