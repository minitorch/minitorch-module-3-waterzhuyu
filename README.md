# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py


Diagnose check results:
```
❯ python project/parallel_check.py
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/home/zuyu/24SP/minitorch_workspace/minitorch-
module-3-waterzhuyu/minitorch/fast_ops.py (154)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /home/zuyu/24SP/minitorch_workspace/minitorch-module-3-waterzhuyu/minitorch/fast_ops.py (154) 
--------------------------------------------------------------------------------------------------------------------------------|loop #ID
    @njit(parallel=True)                                                                                                        | 
    def _map(                                                                                                                   | 
        out: Storage,                                                                                                           | 
        out_shape: Shape,                                                                                                       | 
        out_strides: Strides,                                                                                                   | 
        in_storage: Storage,                                                                                                    | 
        in_shape: Shape,                                                                                                        | 
        in_strides: Strides,                                                                                                    | 
    ) -> None:                                                                                                                  | 
        assert len(out_shape) < MAX_DIMS and len(in_shape) < MAX_DIMS                                                           | 
                                                                                                                                | 
        if (out_strides == in_strides).all() and (out_shape == in_shape).all():  # When `out` and `in` are stride-aligned-------| #0, 1
            for i in prange(out.size):------------------------------------------------------------------------------------------| #2
                out[np.array(i)] = fn(in_storage[np.array(i)])                                                                  | 
            return                                                                                                              | 
                                                                                                                                | 
        out_index: Index = np.zeros_like(out_shape)                                                                             | 
        in_index: Index = np.zeros_like(in_shape)                                                                               | 
                                                                                                                                | 
        for i in prange(out.size):----------------------------------------------------------------------------------------------| #3
            to_index(i, out_shape, out_index)                                                                                   | 
            broadcast_index(out_index, out_shape, in_shape, in_index)                                                           | 
            out[index_to_position(out_index, out_strides)] = fn(in_storage[index_to_position(in_index, in_strides)])            | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #0, #1, #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/home/zuyu/24SP/minitorch_workspace/minitorch-
module-3-waterzhuyu/minitorch/fast_ops.py (203)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /home/zuyu/24SP/minitorch_workspace/minitorch-module-3-waterzhuyu/minitorch/fast_ops.py (203) 
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|loop #ID
    @njit(parallel=True)                                                                                                                                                  | 
    def _zip(                                                                                                                                                             | 
        out: Storage,                                                                                                                                                     | 
        out_shape: Shape,                                                                                                                                                 | 
        out_strides: Strides,                                                                                                                                             | 
        a_storage: Storage,                                                                                                                                               | 
        a_shape: Shape,                                                                                                                                                   | 
        a_strides: Strides,                                                                                                                                               | 
        b_storage: Storage,                                                                                                                                               | 
        b_shape: Shape,                                                                                                                                                   | 
        b_strides: Strides,                                                                                                                                               | 
    ) -> None:                                                                                                                                                            | 
        assert len(out_shape) < MAX_DIMS and len(a_shape) < MAX_DIMS and len(b_shape) < MAX_DIMS                                                                          | 
                                                                                                                                                                          | 
        if (out_strides == a_strides).all() and (out_strides == b_strides).all():-----------------------------------------------------------------------------------------| #4, 5
            for i in prange(out.size):------------------------------------------------------------------------------------------------------------------------------------| #6
                out[np.array(i)] = fn(a_storage[np.array(i)], b_storage[np.array(i)])                                                                                     | 
            return                                                                                                                                                        | 
                                                                                                                                                                          | 
        out_index: Index = np.zeros_like(out_shape)                                                                                                                       | 
        a_index: Index = np.zeros_like(a_shape)                                                                                                                           | 
        b_index: Index = np.zeros_like(b_shape)                                                                                                                           | 
                                                                                                                                                                          | 
        for i in prange(out.size):----------------------------------------------------------------------------------------------------------------------------------------| #7
            to_index(i, out_shape, out_index)                                                                                                                             | 
            broadcast_index(out_index, out_shape, a_shape, a_index)                                                                                                       | 
            broadcast_index(out_index, out_shape, b_shape, b_index)                                                                                                       | 
            out[index_to_position(out_index, out_strides)] = fn(a_storage[index_to_position(a_index, a_strides)], b_storage[index_to_position(b_index, b_strides)])       | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #4, #5, #6, #7).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/home/zuyu/24SP/minitorch_workspace/minitorch-
module-3-waterzhuyu/minitorch/fast_ops.py (254)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /home/zuyu/24SP/minitorch_workspace/minitorch-module-3-waterzhuyu/minitorch/fast_ops.py (254) 
-------------------------------------------------------------------------------------------------|loop #ID
    @njit(parallel=True)                                                                         | 
    def _reduce(                                                                                 | 
        out: Storage,                                                                            | 
        out_shape: Shape,                                                                        | 
        out_strides: Strides,                                                                    | 
        a_storage: Storage,                                                                      | 
        a_shape: Shape,                                                                          | 
        a_strides: Strides,                                                                      | 
        reduce_dim: int,                                                                         | 
    ) -> None:                                                                                   | 
        assert len(out_shape) < MAX_DIMS and len(a_shape) < MAX_DIMS                             | 
                                                                                                 | 
        out_index: Index = np.zeros_like(out_shape)                                              | 
        a_index: Index = np.zeros_like(a_shape)                                                  | 
                                                                                                 | 
        for i in prange(out.size):---------------------------------------------------------------| #8
            to_index(i, out_shape, out_index)                                                    | 
            to_index(i, out_shape, a_index)                                                      | 
                                                                                                 | 
            a_index[reduce_dim] = 0                                                              | 
            reduce_res: np.float64 = a_storage[index_to_position(a_index, a_strides)]            | 
                                                                                                 | 
            for j in range(1, a_shape[reduce_dim]):                                              | 
                a_index[reduce_dim] = j                                                          | 
                reduce_res = fn(reduce_res, a_storage[index_to_position(a_index, a_strides)])    | 
                                                                                                 | 
            out_strides[index_to_position(out_index, out_strides)] = reduce_res                  | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #8).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/home/zuyu/24SP/minitorch_workspace/minitorch-
module-3-waterzhuyu/minitorch/fast_ops.py (285)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /home/zuyu/24SP/minitorch_workspace/minitorch-module-3-waterzhuyu/minitorch/fast_ops.py (285) 
----------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                        | 
    out: Storage,                                                                                   | 
    out_shape: Shape,                                                                               | 
    out_strides: Strides,                                                                           | 
    a_storage: Storage,                                                                             | 
    a_shape: Shape,                                                                                 | 
    a_strides: Strides,                                                                             | 
    b_storage: Storage,                                                                             | 
    b_shape: Shape,                                                                                 | 
    b_strides: Strides,                                                                             | 
) -> None:                                                                                          | 
    """                                                                                             | 
    NUMBA tensor matrix multiply function.                                                          | 
                                                                                                    | 
    Should work for any tensor shapes that broadcast as long as                                     | 
                                                                                                    | 
    ```                                                                                             | 
    assert a_shape[-1] == b_shape[-2]                                                               | 
    ```                                                                                             | 
                                                                                                    | 
    Optimizations:                                                                                  | 
                                                                                                    | 
    * Outer loop in parallel                                                                        | 
    * No index buffers or function calls                                                            | 
    * Inner loop should have no global writes, 1 multiply.                                          | 
                                                                                                    | 
                                                                                                    | 
    Args:                                                                                           | 
        out (Storage): storage for `out` tensor                                                     | 
        out_shape (Shape): shape for `out` tensor                                                   | 
        out_strides (Strides): strides for `out` tensor                                             | 
        a_storage (Storage): storage for `a` tensor                                                 | 
        a_shape (Shape): shape for `a` tensor                                                       | 
        a_strides (Strides): strides for `a` tensor                                                 | 
        b_storage (Storage): storage for `b` tensor                                                 | 
        b_shape (Shape): shape for `b` tensor                                                       | 
        b_strides (Strides): strides for `b` tensor                                                 | 
                                                                                                    | 
    Returns:                                                                                        | 
        None : Fills in `out`                                                                       | 
    """                                                                                             | 
    assert a_shape[-1] == b_shape[-2]                                                               | 
                                                                                                    | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                          | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                          | 
                                                                                                    | 
    # WHY 3 dimension of out_shape? : Cause 2-dim tensor will reshape to (1, *shape)                | 
    for i in prange(out_shape[0]):  # dim of batch--------------------------------------------------| #11
        for j in prange(out_shape[1]):--------------------------------------------------------------| #10
            for k in prange(out_shape[2]):----------------------------------------------------------| #9
                a_inner = i * a_batch_stride + j * a_strides[1]                                     | 
                b_inner = i * b_batch_stride + k * b_strides[2]                                     | 
                                                                                                    | 
                num = 0.                                                                            | 
                for _ in range(a_shape[-1]):  # a_shape[-1] == b_shape[2]                           | 
                    num += a_storage[a_inner] * b_storage[b_inner]                                  | 
                    a_inner += a_strides[2]                                                         | 
                    b_inner += b_strides[1]                                                         | 
                out_pos = np.array(i * out_strides[0] + j * out_strides[1] + k * out_strides[2])    | 
                out[out_pos] = num                                                                  | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #11, #10).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--11 is a parallel loop
   +--10 --> rewritten as a serial loop
      +--9 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--11 (parallel)
   +--10 (parallel)
      +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--11 (parallel)
   +--10 (serial)
      +--9 (serial)


 
Parallel region 0 (loop #11) had 0 loop(s) fused and 2 loop(s) serialized as 
part of the larger parallel loop (#11).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

```