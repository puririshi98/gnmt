#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor stochroundfortensor(torch::Tensor mtx,torch::Tensor half_tensor);
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor stochroundfortensor_cpp(torch::Tensor mtx,torch::Tensor half_tensor){
	CHECK_INPUT(mtx);
	
 

  

	return stochroundfortensor(mtx,half_tensor);
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("stochastic_tensor_round", &stochroundfortensor_cpp, "rounds a float tensor to a half tensor stochastically");
  
}