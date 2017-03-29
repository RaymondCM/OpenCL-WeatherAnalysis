__kernel void min_INT(__global const int *A, __global int *B, __local int *local_min) {
    //Get ID, local ID and width of local workgroup
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);

    local_min[lid] = A[id];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 1; i < N; i *= 2) {
        if (lid % (i * 2) == 0 && lid + i < N) {
            //If next value is less than current, replace current
            if (local_min[lid + i] < local_min[lid])
                local_min[lid] = local_min[lid + i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //Calculate minimum sequentially
    if (lid == 0) {
        atomic_min(&B[0], local_min[lid]);
    }
}

__kernel void min_FLOAT(__global const float *A, __global float *B, __local float *local_min) {
    //Get ID, local ID and width of local workgroup
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);

    local_min[lid] = A[id];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 1; i < N; i *= 2) {
        if (lid % (i * 2) == 0 && lid + i < N) {
            //If next value is less than current, replace current
            if (local_min[lid + i] < local_min[lid])
                local_min[lid] = local_min[lid + i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //Calculate minimum sequentially
    if (lid == 0) {
        B[get_group_id(0)] = local_min[lid];

        barrier(CLK_LOCAL_MEM_FENCE);
        if (id == 0) {

            int group_count = get_num_groups(0);
            for (int i = 1; i < group_count; ++i) {
                if (B[i] < B[id])
                    B[id] = B[i];
            }
        }

    }
}

__kernel void max_INT(__global const int *A, __global int *B, __local int *local_max) {
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);

    local_max[lid] = A[id];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 1; i < N; i *= 2) {
        if (lid % (i * 2) == 0 && lid + i < N) {
            if (local_max[lid + i] > local_max[lid])
                local_max[lid] = local_max[lid + i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_max(&B[0], local_max[lid]);
    }
}

__kernel void max_FLOAT(__global const float *A, __global float *B, __local float *local_max) {
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);

    local_max[lid] = A[id];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 1; i < N; i *= 2) {
        if (lid % (i * 2) == 0 && lid + i < N) {
            if (local_max[lid + i] > local_max[lid])
                local_max[lid] = local_max[lid + i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //Calculate minimum sequentially
    if (lid == 0) {
        B[get_group_id(0)] = local_max[lid];

        barrier(CLK_LOCAL_MEM_FENCE);
        if (id == 0) {

            int group_count = get_num_groups(0);
            for (int i = 1; i < group_count; ++i) {
                if (B[i] > B[id])
                    B[id] = B[i];
            }
        }

    }
}

__kernel void sum_INT(__global const int *A, __global int *B, __local int *local_sum) {
    //Get ID, local ID and width of local workgroup
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);

    local_sum[lid] = A[id];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 1; i < N; i *= 2) {
        if (lid % (i * 2) == 0 && lid + i < N)
            local_sum[lid] += local_sum[lid + i];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //Calculate minimum sequentially
    if (lid == 0) {
        atomic_add(&B[0], local_sum[lid]);
    }
}

__kernel void sum_FLOAT(__global const float *A, __global float *B, __local float *local_sum) {
    //Get ID, local ID and width of local workgroup
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);

    local_sum[lid] = A[id];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 1; i < N; i *= 2) {
        if (lid % (i * 2) == 0 && lid + i < N)
            local_sum[lid] += local_sum[lid + i];

        barrier(CLK_LOCAL_MEM_FENCE);
    }


    //Calculate minimum sequentially
    if (lid == 0) {
        B[get_group_id(0)] = local_sum[lid];

        barrier(CLK_LOCAL_MEM_FENCE);
        if (id == 0) {

            int group_count = get_num_groups(0);
            for (int i = 1; i < group_count; ++i) {
                B[id] += B[i];
            }
        }

    }
}

inline int square_int(int a) {
    return a * a;
}

inline float square_flt(float a) {
    return a * a;
}

__kernel void std_INT(__global const int *A, __global int *B, float mean_f, __local int *local_std) {
    //Get ID, local ID and width of local workgroup
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);
    int mean = (int) mean_f;

    local_std[lid] = square_int(A[id] - mean);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 1; i < N; i *= 2) {
        if (lid % (i * 2) == 0 && lid + i < N)
            local_std[lid] += local_std[lid + i];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //Calculate minimum sequentially
    if (lid == 0) {
        atomic_add(&B[0], local_std[lid]);
    }
}

__kernel void std_manual_INT(__global const int *A, __global int *B, float mean_f, __local int *local_std) {
    //Get ID, local ID and width of local work group
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);
    int mean = (int) mean_f;

    local_std[lid] = square_int(A[id] - mean);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 1; i < N; i *= 2) {
        if (lid % (i * 2) == 0 && lid + i < N)
            local_std[lid] += local_std[lid + i];

        barrier(CLK_LOCAL_MEM_FENCE);
    }


    //Calculate minimum sequentially
    if (lid == 0) {
        B[get_group_id(0)] = local_std[lid];

        barrier(CLK_LOCAL_MEM_FENCE);
        if (id == 0) {

            int group_count = get_num_groups(0);
            for (int i = 1; i < group_count; ++i) {
                B[id] += B[i];
            }
            printf("%d", B[id]);
            B[id] = sqrt((float) B[id] / get_global_size(0));
        }

    }
}

__kernel void std_FLOAT(__global const float *A, __global float *B, float mean, __local float *local_std) {
    //Get ID, local ID and width of local workgroup
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);

    local_std[lid] = square_flt(A[id] - mean);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 1; i < N; i *= 2) {
        if (lid % (i * 2) == 0 && lid + i < N)
            local_std[lid] += local_std[lid + i];

        barrier(CLK_LOCAL_MEM_FENCE);
    }


    //Calculate minimum sequentially
    if (lid == 0) {
        B[get_group_id(0)] = local_std[lid];

        barrier(CLK_LOCAL_MEM_FENCE);
        if (id == 0) {

            int group_count = get_num_groups(0);
            for (int i = 1; i < group_count; ++i) {
                B[id] += B[i];
            }

            B[id] = sqrt(B[id] / get_global_size(0));
        }

    }
}

__kernel void sort_naive_INT(__global const int *A, __global int *B, __local int *local_sorted) {
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);

    local_sorted[lid] = A[id];
    barrier(CLK_LOCAL_MEM_FENCE);

    int sorted_pos = 0;
    for (int j = 0; j < N; j++) {
        if (local_sorted[j] < local_sorted[lid])
            sorted_pos++;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    B[sorted_pos] = local_sorted[lid];
}

__kernel void sort_INT(__global int* A,  __global int* B)
{
    uint id = get_global_id(0);
    local int LA[16];
    int n = get_global_size(0);
    int l = get_local_size(0);
    LA[id] = A[id];
    LA[id + 8] = A[id + 8];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = 1; i <= l; i++) {

        uint signo = (id >> (i - 1)) & 1;
        for (uint j = i; j>0; j--) {

            uint t = 1 << (j-1);

            uint index = (id >> (j-1));
            index = index << j;
            index = index + (id & (t - 1));
            uint index2 = index + t;

            if ((LA[index] > LA[index2]) ^ (signo)) {
                int aux = LA[index];
                LA[index] = LA[index2];
                LA[index2] = aux;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    B[id] = LA[id];
    B[id + 8] = LA[id + 8];


}