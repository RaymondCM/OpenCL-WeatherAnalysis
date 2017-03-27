__kernel void minimum(__global const int* A, __global int* B, __local int* local_min)
{
	//Get ID, local ID and width of local workgroup
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	local_min[lid] = A[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (lid % (i * 2) == 0 && lid + i < N)
		{	
			//If next value is less than current, replace current
			if (local_min[lid+i] < local_min[lid])
				local_min[lid] = local_min[lid+i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//Calculate minimum sequentially
	if (lid == 0)
	{
		atomic_min(&B[0], local_min[lid]);
	}
}

__kernel void maximum(__global const int* A, __global int* B, __local int* local_max)
{
	//Get ID, local ID and width of local workgroup
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	local_max[lid] = A[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (lid % (i * 2) == 0 && lid + i < N)
		{	
			//If next value is less than current, replace current
			if (local_max[lid+i] > local_max[lid])
				local_max[lid] = local_max[lid+i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//Calculate minimum sequentially
	if (lid == 0)
	{
		atomic_max(&B[0], local_max[lid]);
	}
}

__kernel void sum(__global const int *A, __global int *B, __local int *local_sum) {
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

inline int square(int a) {
    return a * a;
}

__kernel void std(__global const int *A, __global int *B, int mean, __local int *local_sum) {
    //Get ID, local ID and width of local workgroup
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);

    local_sum[lid] = A[id];
    barrier(CLK_LOCAL_MEM_FENCE);

    local_sum[lid] = square(local_sum[lid] - mean);
    for (int i = 1; i < N; i *= 2) {
        if (lid % (i * 2) == 0 && lid + i < N)
            local_sum[lid] += square(local_sum[lid + i] - mean);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    //Calculate minimum sequentially
    if (lid == 0) {
        atomic_add(&B[0], local_sum[lid]);
    }
}