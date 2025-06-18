#include "Header.cuh"

// *** *** *** *** ***

__device__ float RandomFloat(unsigned n) {
	const unsigned a = 1664525;
	const unsigned c = 1013904223;

	unsigned tmp1 = 1;
	unsigned tmp2 = a;
	unsigned tmp3 = 0;
	while (n != 0) {
		if ((n & 1) != 0) tmp3 = (tmp2 * tmp3) + tmp1;
		tmp1 = (tmp2 * tmp1) + tmp1;
		tmp2 = tmp2 * tmp2;
		n >>= 1;
	}
	float result = __uint_as_float(1065353216 | ((tmp3 * c) & 8388607)) - 1.0f;
	return result;
}

// *** *** *** *** ***

__global__ void SampleBoxUniform(unsigned n, unsigned nStart, float4 lower_bound, float4 upper_bound, float4 *samples) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < n) {
		float U1 = RandomFloat(nStart + (tid << 2));
		float U2 = RandomFloat(nStart + (tid << 2) + 1);
		float U3 = RandomFloat(nStart + (tid << 2) + 2);
		float U4 = RandomFloat(nStart + (tid << 2) + 3);
		float4 sample = make_float4(
			lower_bound.x + ((upper_bound.x - lower_bound.x) * U1),
			lower_bound.y + ((upper_bound.y - lower_bound.y) * U2),
			lower_bound.z + ((upper_bound.z - lower_bound.z) * U3),
			lower_bound.w + ((upper_bound.w - lower_bound.w) * U4)
		);
		samples[tid] = sample;
	}
}

// *** *** *** *** ***

const int NUMBER_OF_MEANS = 1024 * 1024 * 1;
const int NUMBER_OF_QUERIED_POINTS = 1024 * 1024 * 32;
const int K = 16;

const float4 lower_bound_means = make_float4(-1.0f, -1.0f, -1.0f, 0.0f);
const float4 upper_bound_means = make_float4(1.0f, 1.0f, 1.0f, 0.01f);

const float4 lower_bound_queried_points = make_float4(-1.0f, -1.0f, -1.0f, 0.0f);
const float4 upper_bound_queried_points = make_float4(1.0f, 1.0f, 1.0f, 0.0f);

float chi_square_squared_radius = 11.3449f;

// *** *** *** *** ***

int main() {
	cudaError_t error_CUDA;

	// *** *** *** *** ***

	error_CUDA = cudaSetDevice(0);
	if (error_CUDA != cudaSuccess) return false;

	// *** *** *** *** ***

	unsigned n = 0;

	// *** *** *** *** ***

	float4 *means;

	error_CUDA = cudaMalloc(&means, sizeof(float4) * NUMBER_OF_MEANS);
	if (error_CUDA != cudaSuccess) return false;

	SampleBoxUniform<<<(NUMBER_OF_MEANS + 63) >> 6, 64>>>(NUMBER_OF_MEANS, n, lower_bound_means, upper_bound_means, means);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) return false;

	n += (NUMBER_OF_MEANS << 2); // !!! !!! !!!

	// *** *** *** *** ***

	float4 *queried_points;

	error_CUDA = cudaMalloc(&queried_points, sizeof(float4) * NUMBER_OF_QUERIED_POINTS);
	if (error_CUDA != cudaSuccess) return false;

	SampleBoxUniform<<<(NUMBER_OF_QUERIED_POINTS + 63) >> 6, 64>>>(
		NUMBER_OF_QUERIED_POINTS,
		n,
		lower_bound_queried_points,
		upper_bound_queried_points,
		queried_points
		);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) return false;

	n += (NUMBER_OF_QUERIED_POINTS << 2); // !!! !!! !!!

	// *** *** *** *** ***

	S_CUDA_KNN cknn;
	bool result;

	// *** *** *** *** ***

	result = CUDA_KNN_Init(chi_square_squared_radius, &cknn);
	if (!result) {
		printf("CUDA_KNN_Init failed... .\n");
		return false;
	} else
		printf("CUDA_KNN_Init succeeded... .\n");



	while(true) {

		// *** *** *** *** ***

		result = CUDA_KNN_Fit(means, NUMBER_OF_MEANS, &cknn);
		if (!result) {
			printf("CUDA_KNN_Fit failed... .\n");
			return false;
		} else
			printf("CUDA_KNN_Fit succeeded... .\n");

		printf("Done... .\n");

		// *** *** *** *** ***

		int passNum = 1;

		float *distances;
		int *indices;

		error_CUDA = cudaMalloc(&distances, sizeof(float) * NUMBER_OF_QUERIED_POINTS * K);
		if (error_CUDA != cudaSuccess) return false;

		error_CUDA = cudaMalloc(&indices, sizeof(int) * NUMBER_OF_QUERIED_POINTS * K);
		if (error_CUDA != cudaSuccess) return false;

		result = CUDA_KNN_KNeighbors(
			queried_points,
			NUMBER_OF_QUERIED_POINTS,
			K,
			distances,
			indices,
			&cknn
		);
		if (!result) {
			printf("[%d]: CUDA_KNN_KNeighbors failed... .\n", passNum);
			return false;
		} else
			printf("[%d]: CUDA_KNN_KNeighbors succeeded... .\n", passNum);

		// *** *** *** *** ***

		// TEST (massive slowdown due to the Device->Host memory copy)
		float *distances_host = (float *)malloc(sizeof(float) * NUMBER_OF_QUERIED_POINTS * K);
		int *indices_host = (int *)malloc(sizeof(int) * NUMBER_OF_QUERIED_POINTS * K);
		float4 *means_host = (float4 *)malloc(sizeof(float4) * NUMBER_OF_MEANS);
		float4 *queried_points_host = (float4 *)malloc(sizeof(float4) * NUMBER_OF_QUERIED_POINTS);

		error_CUDA = cudaMemcpy(distances_host, distances, sizeof(float) * NUMBER_OF_QUERIED_POINTS * K, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) return false;
		
		error_CUDA = cudaMemcpy(indices_host, indices, sizeof(int) * NUMBER_OF_QUERIED_POINTS * K, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) return false;

		error_CUDA = cudaMemcpy(means_host, means, sizeof(float4) * NUMBER_OF_MEANS, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) return false;

		error_CUDA = cudaMemcpy(queried_points_host, queried_points, sizeof(float4) * NUMBER_OF_QUERIED_POINTS, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) return false;
		
		for (int j = 0; j < K; ++j) {
			float distance = distances_host[(j * NUMBER_OF_QUERIED_POINTS) + 23];
			int index = indices_host[(j * NUMBER_OF_QUERIED_POINTS) + 23];
			printf("POINT: %d; RANK: %d; DISTANCE: %f; INDEX: %d;\n", 0 + 1, j + 1, distance, index);
		}

		printf("\n");

		float *avg_distance = (float *)malloc(sizeof(float) * K);

		for (int i = 0; i < K; ++i) avg_distance[i] = INFINITY;
		for (int i = 0; i < 128; ++i) {
			float4 point = queried_points_host[i];
			float dist_min_prev = -INFINITY;
			int ind_min_prev = -1;

			for (int j = 0; j < K; ++j) {
				float dist_min = INFINITY;
				int ind_min = NUMBER_OF_QUERIED_POINTS;

				for (int k = 0; k < NUMBER_OF_MEANS; ++k) {
					float4 mean = means_host[k];
					float3 d = make_float3(mean.x - point.x, mean.y - point.y, mean.z - point.z);
					float dist = sqrtf((d.x * d.x) + (d.y * d.y) + (d.z * d.z));
					if (dist < dist_min) {
						if ((dist > dist_min_prev) || (((dist == dist_min_prev) && (k > ind_min_prev)))) {
							dist_min = dist;
							ind_min = k;
						}
					} else {
						if ((dist == dist_min) && (k < ind_min)) {
							if ((dist > dist_min_prev) || (((dist == dist_min_prev) && (k > ind_min_prev)))) {
								dist_min = dist;
								ind_min = k;
							}
						}
					}
				}

				float dist_approx = distances_host[(j * NUMBER_OF_QUERIED_POINTS) + i];
				int ind = indices_host[(j * NUMBER_OF_QUERIED_POINTS) + i];
				float tmp = fabsf(dist_approx - dist_min);
				if (isfinite(tmp)) {
					if (!isfinite(avg_distance[j])) avg_distance[j] = tmp;
					else
						avg_distance[j] += tmp;
				}

				dist_min_prev = dist_min;
				ind_min_prev = ind_min;
			}
		}
		for (int i = 0; i < K; ++i) avg_distance[i] /= K;
		for (int i = 0; i < K; ++i) printf("%d: %f\n", i + 1, avg_distance[i]);

		free(distances_host);
		free(indices_host);
		free(means_host);
		free(queried_points_host);
		free(avg_distance);

		// *** *** *** *** ***

		error_CUDA = cudaFree(distances);
		if (error_CUDA != cudaSuccess) return false;

		error_CUDA = cudaFree(indices);
		if (error_CUDA != cudaSuccess) return false;

		// *** *** *** *** ***

		++passNum;
	}

	error_CUDA = cudaFree(means);
	if (error_CUDA != cudaSuccess) return false;

	error_CUDA = cudaFree(queried_points);
	if (error_CUDA != cudaSuccess) return false;
}