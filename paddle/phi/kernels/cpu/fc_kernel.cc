
#include "paddle/phi/kernels/impl/fc_kernel_impl.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

PD_REGISTER_KERNEL(fc, GPU, ALL_LAYOUT, phi::FcKernel, float, double) {
  //   if (kernel_key.dtype() == phi::DataType::INT8) {
  //     kernel->OutputAt(0).SetDataType(phi::DataType::INT32);
  //   }
}
