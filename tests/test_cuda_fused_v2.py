"""Fused CUDA summaries preserve the validated traversal/counting behavior."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def simulated(code):
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    env["NUMBA_NUM_THREADS"] = "4"
    completed = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env,
                               text=True, capture_output=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_fused_counts_match_original_combined_and_reduction_kernels():
    simulated("""
        import numpy as np
        from numba import cuda
        from raystrack.utils.prepared import PreparedSolver
        from raystrack.utils.cuda_trace import (kernel_trace_combined,
            kernel_trace_bvh_combined, kernel_trace_instanced_combined,
            kernel_reduce_hits, kernel_bin_tregenza, kernel_count_upward_misses)
        from raystrack.backends.cuda_fused import (kernel_fused_flat,
            kernel_fused_bvh, kernel_fused_instanced)

        def square(name,z,half,down=False):
            v=np.asarray([[-half,-half,z],[half,-half,z],[half,half,z],[-half,half,z]],np.float32)
            f=np.asarray([[0,1,2],[0,2,3]],np.int32)
            return name,v,f[:,::-1].copy() if down else f
        meshes=[square('emitter',0,1),square('blocker',1,.4,True),square('receiver',2,1.2)]
        rng=np.random.default_rng(781)
        orig=np.asarray(rng.uniform(-.6,.6,(37,3)),np.float32);orig[:,2]=.0001
        dirs=np.asarray(rng.normal(size=(37,3)),np.float32)
        dirs/=np.linalg.norm(dirs,axis=1)[:,None]
        dirs[:5]=np.asarray([[0,0,1],[0,0,-1],[1,0,0],[.8,0,.6],[0,.8,.6]],np.float32)
        origins,directions=cuda.to_device(orig),cuda.to_device(dirs)
        checks=0
        for acceleration,use_bvh in (('flat',False),('flat',True),('instanced',True)):
            prepared=PreparedSolver(meshes,acceleration=acceleration)
            scene=prepared.get_instanced_scene() if acceleration=='instanced' else prepared.get_scene(use_bvh=use_bvh)
            ds=prepared.get_device_instanced_scene() if acceleration=='instanced' else prepared.get_device_scene(use_bvh=use_bvh)
            for surf_count in (3,1025):
                active=np.ones(surf_count,np.uint8)
                # Both fully enabled and masked geometry must preserve occlusion.
                for masked in (False,True):
                    active[1]=not masked
                    device_active=cuda.to_device(active)
                    hit=cuda.device_array(len(orig),np.int32);front=cuda.device_array(len(orig),np.uint8);mask=cuda.device_array(len(orig),np.uint8)
                    if acceleration=='instanced':
                        kernel_trace_instanced_combined[3,16](origins,directions,*ds,device_active,0,0,hit,front,mask)
                    else:
                        args=(origins,directions,ds.v0,ds.e1,ds.e2,ds.normals,ds.sid,device_active)
                        if use_bvh:
                            kernel_trace_bvh_combined[3,16](*args,ds.bb_min,ds.bb_max,ds.left,ds.right,ds.start,ds.count,0,0,hit,front,mask)
                        else:
                            kernel_trace_combined[3,16](*args,0,0,hit,front,mask)
                    for discrete,include_sky in ((False,False),(False,True),(True,True)):
                        width=2*surf_count+(145 if discrete else 1)
                        base=np.zeros(width,np.int64)
                        # Global totals remain int64 even though block counts use int32.
                        base[1]=2**31-2;base[-1]=2**31+1
                        expected=cuda.to_device(base);actual=cuda.to_device(base)
                        kernel_reduce_hits[3,16](hit,front,expected[:surf_count],expected[surf_count:2*surf_count],surf_count)
                        if include_sky:
                            if discrete: kernel_bin_tregenza[3,16](directions,mask,expected[2*surf_count:])
                            else: kernel_count_upward_misses[3,16](directions,mask,expected[2*surf_count:])
                        if acceleration=='instanced':
                            kernel_fused_instanced[3,16](origins,directions,*ds,device_active,0,surf_count,discrete,include_sky,actual)
                        elif use_bvh:
                            kernel_fused_bvh[3,16](*args,ds.bb_min,ds.bb_max,ds.left,ds.right,ds.start,ds.count,0,surf_count,discrete,include_sky,actual)
                        else:
                            kernel_fused_flat[3,16](*args,0,surf_count,discrete,include_sky,actual)
                        np.testing.assert_array_equal(actual.copy_to_host(),expected.copy_to_host())
                        checks+=1
        assert checks==36
        print('36 fused CUDA flat/BVH/instance, active-mask, large-histogram and sky parity checks passed')
    """)


def test_workspace_keeps_gpu_ray_generation_and_one_compact_replicate_readback():
    simulated("""
        from unittest import mock
        import numpy as np
        from raystrack.utils.prepared import PreparedSolver
        from raystrack.backends.workspaces import _CudaWorkspace,_CpuWorkspace

        v=np.asarray([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],np.float32)
        f=np.asarray([[0,1,2],[0,2,3]],np.int32)
        meshes=[('source',v,f),('target',v+[0,0,1],f[:,::-1].copy())]
        specs=[((.1,.2),(.3,.4,.5,.6,.7),19,0),((.6,.5),(.4,.3,.2,.1,.9),13,7)]
        for acceleration,use_bvh in (('flat',False),('flat',True),('instanced',True)):
            prepared=PreparedSolver(meshes,acceleration=acceleration)
            scene=prepared.get_instanced_scene() if acceleration=='instanced' else prepared.get_scene(use_bvh=use_bvh)
            emitter=prepared.get_emitters(samples=2,rays=4,flip_faces=False)[0]
            for async_,gpu_raygen in ((False,False),(True,True)):
                workspace=_CudaWorkspace(async_)
                options=dict(samples=2,rays=4,flip_faces=False,discrete=True,include_sky=True,
                    surf_active=np.ones(2,np.uint8),gpu_raygen=gpu_raygen)
                actual=workspace.trace_many(prepared,0,scene,emitter,2,specs,**options)
                cpu=_CpuWorkspace()
                expected=cpu.trace_many(prepared,0,scene,emitter,2,specs,emit_sid=0,**options)
                for got,want in zip(actual,expected):
                    for left,right in zip(got,want): np.testing.assert_array_equal(left,right)
                summary=workspace.summary
                array_type=type(summary);original=array_type.copy_to_host
                reads=[]
                def record(array,*args,**kwargs):
                    reads.append(array.shape)
                    return original(array,*args,**kwargs)
                with mock.patch.object(array_type,'copy_to_host',record):
                    repeated=workspace.trace_many(prepared,0,scene,emitter,2,specs,**options)
                assert reads==[(2,149)],reads
                assert workspace.summary is summary
                for old,new in zip(actual,repeated):
                    for left,right in zip(old,new): np.testing.assert_array_equal(left,right)
        print('GPU/CPU ray generation, compact batched copies and buffer reuse passed')
    """)


def test_common_workspace_cache_and_portable_replicate_protocol():
    from types import SimpleNamespace
    from unittest import mock
    import numpy as np
    from raystrack.backends.workspaces import workspace_for

    prepared, cfg = SimpleNamespace(), SimpleNamespace(cuda_async=True)
    cpu = workspace_for(prepared, "cpu", cfg)
    assert workspace_for(prepared, "cpu", cfg) is cpu
    assert cpu.trace_many(prepared, 0, None, None, 2, []) == []
    backend = mock.Mock()
    backend.trace_iterations.return_value = (np.asarray([[1, 2], [3, 4]], np.int64),
                                            np.zeros((2, 2), np.int64), np.zeros((2, 145), np.int64))
    backend.trace_batch.return_value = (np.asarray([1, 2], np.int64), np.zeros(2, np.int64), np.zeros(145, np.int64))
    options = dict(samples=2, rays=4, flip_faces=False, emit_sid=0,
                   discrete=True, include_sky=True, gpu_raygen=True,
                   surf_active=np.ones(2, np.uint8))
    specs = [((.1, .2), (.3, .4, .5, .6, .7), 8, 0),
             ((.6, .5), (.4, .3, .2, .1, .9), 8, 0)]
    with mock.patch("raystrack.utils.taichi_trace.get_taichi_backend", return_value=backend) as getter:
        portable = workspace_for(prepared, "vulkan", cfg)
        assert workspace_for(prepared, "vulkan", cfg) is portable
        assert getter.call_count == 1
        actual = portable.trace_many(prepared, 0, None, None, 2, specs, **options)
        assert len(actual) == 2
        np.testing.assert_array_equal(actual[1][0], [3, 4])
        grouped = backend.trace_iterations.call_args.kwargs
        assert "samples" not in grouped and "flip_faces" not in grouped
        assert grouped["ray_count"] == 8 and grouped["ray_offset"] == 0
        backend.trace_batch.assert_not_called()
        different = [specs[0], (*specs[1][:2], 3, 5)]
        actual = portable.trace_many(prepared, 0, None, None, 2, different, **options)
        assert len(actual) == 2 and backend.trace_batch.call_count == 2
        assert backend.trace_batch.call_args.kwargs["ray_count"] == 3
        assert backend.trace_batch.call_args.kwargs["ray_offset"] == 5
        assert portable.trace_many(prepared, 0, None, None, 2, [], **options) == []
