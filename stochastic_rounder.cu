#include <torch/extension.h>
#include <curand_kernel.h>
#include <curand.h>
#include <curand_mtgp32_host.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>
#include <curand_mtgp32dc_p_11213.h>
#include "philox_random.h"
#include "philox_pytorch.h"
#include <stdio.h>      
#include <stdlib.h>
#include <time.h>
using namespace std;

static uint64_t offset=0;
// float holdy1=pow(2.0,-10.0);
// float holdy2=pow(2.0,-24.0);
__device__ const float twoten=0.0009765625;
__device__ const float twominustwentyfour=0.000000059604644775390625;

template<typename T>
__device__ __forceinline__ T maybe_upcast(__half x){ return T(__half2float(x)); }
template<> __device__ __forceinline__ __half maybe_upcast<__half>(__half x){ return x; }

__device__ __forceinline__ float get_delta_fp16(float x){
	int e_actual;
	frexpf(x, &e_actual);
	e_actual-=1;
	// int e_actual=e_stored-127;
	if(e_actual>=-14){
		return twoten*pow(2,e_actual);
	}
	else{
		return twominustwentyfour;
	}
}
template <typename scalar_t>
__device__ __forceinline__ scalar_t natalia_magic(float x,curandStatePhilox4_32_10_t state){
 	if(x==0.0){

 		return scalar_t(0.0);
 	}
	float delta=get_delta_fp16(x);
	
	float randy=curand_uniform(&state);
	float val;
	if(x<0.0){
	    val=x-randy*delta;
	}
	else{
	    val=x+randy*delta;
	}
	// To guarantee representability, route through a guaranteed FP16 cast.
	return maybe_upcast<scalar_t>(__float2half_rz(val));
}

template <typename scalar_t>
__global__ void stochround(float* mtx,scalar_t* new_mtx, int n, uint64_t seed, uint64_t offset){
	
	int threadnum=blockDim.x*blockIdx.x+threadIdx.x;
	curandStatePhilox4_32_10_t state;
	curand_init(seed,threadnum,offset,&state);
	for(int i = threadnum; i <n ; i +=blockDim.x*gridDim.x ){
		float mtx_holder=static_cast<float>(mtx[i]);
		new_mtx[i]=natalia_magic<scalar_t>(mtx_holder,state);
	}

}
torch::Tensor stochroundfortensor(torch::Tensor mtx,torch::Tensor half_mtx){
	torch::IntArrayRef sizes=mtx.sizes();
	int dims=sizes.size();
	size_t n = 1;
	for(int county=0;county<dims;county++){
		n=n*sizes[county];
	}
	
	

	uint64_t seed= 12345ul;
	
	
	
	const int threads = 256.0;
	
	// printf("%d \n \n \n \n ",offset);
	
	float sm_max=72.0;
	float numthreads_per_sm=1024.0;
	const dim3 blocks(ceil(sm_max*numthreads_per_sm/threads),1,1);
	AT_DISPATCH_FLOATING_TYPES_AND_HALF(half_mtx.scalar_type(),"stochastic_tensor_round",([&] {stochround<scalar_t><<<blocks, threads>>>(mtx.data<float>(),half_mtx.data<scalar_t>(),n,seed,offset);}));
	offset = offset + (n + blocks.x*threads - 1)/(blocks.x*threads);
	// printf("%d \n \n \n \n ",offset);
	return half_mtx;
}

