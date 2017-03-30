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

__kernel void min_WG_REDUCE_FLOAT(__global const float *A, __global float *B, __local float *local_min) {
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

__kernel void max_WG_REDUCE_FLOAT(__global const float *A, __global float *B, __local float *local_max) {
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

__kernel void sum_WG_REDUCE_FLOAT(__global const float *A, __global float *B, __local float *local_sum) {
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

            B[0] = sqrt(B[id] / (get_global_size(0) - 1));
        }

    }
}

__kernel void sort_INT(__global const int* in, __global int* out, __local int* scratch, int merge)
{
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int gid = get_group_id(0);
    int N = get_local_size(0);

    int max_group = (get_global_size(0) / N) - 1;
    int offset_id = id + ((N/2) * merge);

    if (merge && gid == 0)
    {
        out[id] = in[id];
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    scratch[lid] = in[offset_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int l=1; l<N; l<<=1)
    {
        bool direction = ((lid & (l<<1)) != 0);

        for (int inc=l; inc>0; inc>>=1)
        {
            int j = lid ^ inc;
            int i_data = scratch[lid];
            int j_data = scratch[j];

            bool smaller = (j_data < i_data) || ( j_data == i_data && j < lid);
            bool swap = smaller ^ (j < lid) ^ direction;

            barrier(CLK_LOCAL_MEM_FENCE);

            scratch[lid] = (swap) ? j_data : i_data;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    out[offset_id] = scratch[lid];
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (merge && gid == max_group)
        out[offset_id] = in[offset_id];
}

__kernel void sort_test_INT(__global const int* in, __global int* out, __local int* scratch, int merge)
{
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int group_id = get_group_id(0);
    int N = get_local_size(0);
    int max_group = (get_global_size(0) - (N/2));

    if(merge) {
        if(id >= N/2 && id <= max_group) {
            int t = out[id];
            int nr = N/2;
            for(int i = 0; i < nr; ++i) {
                if(out[id+i] > out[id+i+N]) {
                    int t = out[id+i+N];
                    out[id+i+N] = out[id+i];
                    out[id+i] = t;
                }
            }
        }
    } else {
        int offset = get_group_id(0) * N;
        in += offset; out += offset;

        scratch[lid] = in[lid];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int length=1;length<N;length<<=1)
        {
            int iKey = scratch[lid];
            int ii = lid & (length-1);  // index in our sequence in 0..length-1
            int sibling = (lid - ii) ^ length; // beginning of the sibling sequence
            int pos = 0;

            for (int inc=length;inc>0;inc>>=1) // increment for dichotomic search
            {
                int j = sibling+pos+inc-1;
                int jKey = scratch[j];
                bool smaller = (jKey < iKey) || ( jKey == iKey && j < lid );
                pos += (smaller)?inc:0;
                pos = min(pos,length);
            }

            int bits = 2*length-1; // mask for destination
            int dest = ((ii + pos) & bits) | (lid & ~bits); // destination index in merged sequence

            barrier(CLK_LOCAL_MEM_FENCE);
            scratch[dest] = scratch[lid];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Write output
        out[lid] = scratch[lid];
    }
}
