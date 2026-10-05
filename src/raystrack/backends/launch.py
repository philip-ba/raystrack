from __future__ import annotations
from typing import Optional, Tuple

def _compute_cuda_launch(n_items: int, preferred_threads: Optional[int], min_blocks: int = 4) -> Tuple[int, int]:
    n = int(max(0, n_items))
    if n <= 0:
        return 1, 1
    from numba import cuda, config
    device = cuda.get_current_device() if not config.ENABLE_CUDASIM else None
    warp = 32
    max_threads = int(device.MAX_THREADS_PER_BLOCK) if device is not None else 256
    threads = int(max(1, min(preferred_threads or 256, max_threads)))
    if threads >= warp:
        threads = ((threads + warp - 1) // warp) * warp
    if n < threads:
        if n >= warp:
            target = max(warp, (n + min_blocks - 1) // min_blocks)
            threads = min(threads, ((target + warp - 1) // warp) * warp)
        else:
            threads = min(threads, max(1, min(n, (n + min_blocks - 1) // min_blocks)))
    threads = max(1, min(threads, max_threads))
    return max(1, (n + threads - 1) // threads), threads
