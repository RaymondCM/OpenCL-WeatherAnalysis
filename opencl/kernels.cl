//Raymond Kirk - 14474219@students.lincoln.ac.uk

//OpenCL Kernel Code
// Organised into three main distinct kernel types:
//		*_INT				- Integer kernel that calculates using type int and atomic functions
//		*_FLOAT				- Float kernel that reduces partial results to single result in the kernel
//		*_WG_REDUCE_FLOAT	- Kernel that is reduced on the host side with multiple kernel calls (Slower)
//Main pattern used is reduction and comments are provided for specific features of each function only, not repeating ones.

__kernel void min_INT(__global const int *A, __global int *B, __local int *local_min) {
    //Get ID, local ID and width of local workgroup
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);

	//Allocate memory from global to local for faster access
    local_min[lid] = A[id];
	//Syncronise workgroups before next stage
    barrier(CLK_LOCAL_MEM_FENCE);

	//Logically check each variable in the workgroup over stride value of i then i * 2
	//Store lowest value in the first element 0 of the local memory
    for (int i = 1; i < N; i *= 2) {
        if (lid % (i * 2) == 0 && lid + i < N) {
            //If next value is less than current, replace current
            if (local_min[lid + i] < local_min[lid])
                local_min[lid] = local_min[lid + i];
        }

		//Syncronise all local workgroups
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //Calculate minimum sequentially using atomic add which reduces the value to a singular one
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

    //Calculate minimum sequentially using reduction on a single unit
    if (lid == 0) {
		//Store all values computed in the workgroup number of B
		//Logically stores them from 0 to N groups
        B[get_group_id(0)] = local_min[lid];
        barrier(CLK_LOCAL_MEM_FENCE);

        if (id == 0) {
			//Check over all of the local computed values in one cu to reduce to a single value
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

    //Calculate minimum sequentially by reducing the computed values into the group id of the workgroup
	//Will be reduced on the host side to eventually only set B[]0] as the summation of all computed workgroups
    if (lid == 0) {
        B[get_group_id(0)] = local_min[lid];
    }
}

//max_INT, max_FLOAT, max_WG_REDUCE_FLOAT, sum_INT, sum_FLOAT, sum_WG_REDUCE_FLOAT
//	all share the same logic as above see comments.
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

    if (lid == 0) {
        B[get_group_id(0)] = local_max[lid];
    }
}

__kernel void sum_INT(__global const int *A, __global int *B, __local int *local_sum) {
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

    if (lid == 0) {
        atomic_add(&B[0], local_sum[lid]);
    }
}

__kernel void sum_FLOAT(__global const float *A, __global float *B, __local float *local_sum) {
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

    if (lid == 0) {
        B[get_group_id(0)] = local_sum[lid];
    }
}

//Utilty functions to square integers and float
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

	//Convert the mean to a float value as it is not an int in the host code.
    int mean = (int) mean_f;

	//Move squared difference global memory to local for summation (Sum reduce logic_ 
    local_std[lid] = square_int(A[id] - mean);
    barrier(CLK_LOCAL_MEM_FENCE);

	//Sum squared differences
    for (int i = 1; i < N; i *= 2) {
        if (lid % (i * 2) == 0 && lid + i < N)
            local_std[lid] += local_std[lid + i];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(&B[0], local_std[lid]);
    }
}

__kernel void std_manual_INT(__global const int *A, __global int *B, float mean_f, __local int *local_std) {
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


    if (lid == 0) {
        B[get_group_id(0)] = local_std[lid];

        barrier(CLK_LOCAL_MEM_FENCE);
        if (id == 0) {

            int group_count = get_num_groups(0);
            for (int i = 1; i < group_count; ++i) {
                B[id] += B[i];
            }

            B[id] = sqrt((float) B[id] / get_global_size(0));
        }

    }
}

__kernel void std_FLOAT(__global const float *A, __global float *B, float mean, __local float *local_std) {
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

    if (lid == 0) {
        B[get_group_id(0)] = local_std[lid];

        barrier(CLK_LOCAL_MEM_FENCE);
        if (id == 0) {

            int group_count = get_num_groups(0);
            for (int i = 1; i < group_count; ++i) {
                B[id] += B[i];
            }

            B[id] = sqrt(B[id] / (get_global_size(0) - 1));
        }
    }
}

//Sorting kernel - Uses a mixture between bitonic sort scan and bucketing the values
//	Sorts all of the values in the workgroups in acsending order then when kernel
//	is next execurted the merge flag is flipped so that it will now compute the
//  inner range sort +N/2 and -N/2 from global width
__kernel void sort_INT(__global const int* in, __global int* out, __local int* scratch, int merge)
{
	//Get global variables
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int gid = get_group_id(0);
    int N = get_local_size(0);

	//Get the bounding sides for the inner range
    int max_group = (get_global_size(0) / N) - 1;
    int offset_id = id + ((N/2) * merge);

	//If the values are in the outer range edges then assign the global memory at that location to them
    if (merge && gid == 0)
    {
        out[id] = in[id];
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

	//Store global memory locally (offset by n/2 * merge)
    scratch[lid] = in[offset_id];
    barrier(CLK_LOCAL_MEM_FENCE);

	//Reduce over strides until stride is N
    for (int l=1; l<N; l<<=1)
    {
        bool direction = ((lid & (l<<1)) != 0);

		//Reduce over strides until stride is N
        for (int inc=l; inc>0; inc>>=1)
        {
			//Get local postion
            int j = lid ^ inc;

			//Store data in variables
            int i_data = scratch[lid];
            int j_data = scratch[j];

			//Calculate if it is smaller, if so swap the values.
            bool smaller = (j_data < i_data) || ( j_data == i_data && j < lid);
            bool swap = smaller ^ (j < lid) ^ direction;

			//Syncronise local memory
            barrier(CLK_LOCAL_MEM_FENCE);
			//Allocate correctly swapped values
            scratch[lid] = (swap) ? j_data : i_data;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

	//Write to output buffer
    out[offset_id] = scratch[lid];
    barrier(CLK_GLOBAL_MEM_FENCE);

	//If on edge bound 
    if (merge && gid == max_group)
        out[offset_id] = in[offset_id];
}

__kernel void sort_FLOAT(__global const float* in, __global float* out, __local float* scratch, int merge)
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
            float i_data = scratch[lid];
            float j_data = scratch[j];

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
