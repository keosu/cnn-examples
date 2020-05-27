# some inference framworks

# MNN

## install package structure
```
    |-  include/MNN/MNNDefine.h
    |-  include/MNN/Interpreter.hpp
    |-  include/MNN/HalideRuntime.h
    |-  include/MNN/Tensor.hpp
    |-  include/MNN/ErrorCode.hpp
    |-  include/MNN/ImageProcess.hpp
    |-  include/MNN/Matrix.h
    |-  include/MNN/Rect.h
    |-  include/MNN/MNNForwardType.h
    |-  include/MNN/AutoTime.hpp
    |-  include/MNN/expr/Expr.hpp
    |-  include/MNN/expr/ExprCreator.hpp
    |-  include/MNN/expr/MathOp.hpp
    |-  include/MNN/expr/NeuralNetWorkOp.hpp
    |-  include/MNN/expr/Optimizer.hpp
    |-  include/MNN/expr/Executor.hpp
    |-  lib/libMNN.so
```


## convert
| MNNConvert -f TF --modelFile model-mobilenet_v1_075.pb --MNNModel pose.mnn --bizCode biz  

Dynamic section at offset 0x14fce8 contains 30 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [ld-linux-x86-64.so.2]

## 示例
```c++

    // create net and session
    auto mnnNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(poseModel));
    MNN::ScheduleConfig netConfig;
    netConfig.type      = MNN_FORWARD_CPU;
    netConfig.numThread = 4;
    auto session        = mnnNet->createSession(netConfig);

    // set input
    auto input = mnnNet->getSessionInput(session, nullptr);
    // feed data

    //run 
    mnnNet->runSession(session); 

    // get output
    auto output         = mnnNet->getSessionOutput(session, NODE_NAME); 
 
    Tensor displacementFwdHost(output, Tensor::CAFFE); 
    output->copyToHostTensor(&displacementFwdHost); 
 
```

# MACE

```

    |-  include/mace
    |-  include/mace/port
    |-  include/mace/port/env.h
    |-  include/mace/port/file_system.h
    |-  include/mace/port/logger.h
    |-  include/mace/port/port-arch.h
    |-  include/mace/port/port.h
    |-  include/mace/public
    |-  include/mace/public/mace.h
    |-  include/mace/utils
    |-  include/mace/utils/logging.h
    |-  include/mace/utils/macros.h
    |-  include/mace/utils/memory.h
    |-  include/mace/utils/string_util.h
    |-  lib/libprotobuf-lite.a
    |-  lib/rpcmem.a
    |-  lib/libgenerated_version.a
    |-  lib/libgenerated_opencl_kernel.a
    |-  lib/libmodel.a
    |-  lib/libmodel_shared.so
    |-  lib/libcore.a
    |-  lib/libmace.so
    |-  lib/libmace_static.a
    |-  lib/libops.a
    |-  lib/libport_base.a
    |-  lib/libport_posix.a
    |-  lib/libport_linux_base.a
    |-  lib/libport_linux.a
    |-  lib/libproto.a
    |-  lib/libutils.a
    |-  bin/mace_run

```


Dynamic section at offset 0x225d10 contains 32 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
 0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [ld-linux-x86-64.so.2]

 ```c++

 // 添加头文件按
#include "mace/public/mace.h"

// 0. 指定目标设备
DeviceType device_type = DeviceType::GPU;

// 1. 运行配置
MaceStatus status;
MaceEngineConfig config(device_type);
std::shared_ptr<GPUContext> gpu_context;
// Set the path to store compiled OpenCL kernel binaries.
// please make sure your application have read/write rights of the directory.
// this is used to reduce the initialization time since the compiling is too slow.
// It's suggested to set this even when pre-compiled OpenCL program file is provided
// because the OpenCL version upgrade may also leads to kernel recompilations.
const std::string storage_path ="path/to/storage";
gpu_context = GPUContextBuilder()
    .SetStoragePath(storage_path)
    .Finalize();
config.SetGPUContext(gpu_context);
config.SetGPUHints(
    static_cast<GPUPerfHint>(GPUPerfHint::PERF_NORMAL),
    static_cast<GPUPriorityHint>(GPUPriorityHint::PRIORITY_LOW));

// 2. 指定输入输出节点
std::vector<std::string> input_names = {...};
std::vector<std::string> output_names = {...};

// 3. 创建引擎实例
std::shared_ptr<mace::MaceEngine> engine;
MaceStatus create_engine_status;

create_engine_status =
    CreateMaceEngineFromProto(model_graph_proto,
                              model_graph_proto_size,
                              model_weights_data,
                              model_weights_data_size,
                              input_names,
                              output_names,
                              device_type,
                              &engine);
if (create_engine_status != MaceStatus::MACE_SUCCESS) {
  // fall back to other strategy.
}

// 4. 创建输入输出缓存
std::map<std::string, mace::MaceTensor> inputs;
std::map<std::string, mace::MaceTensor> outputs;
for (size_t i = 0; i < input_count; ++i) {
  // Allocate input and output
  int64_t input_size =
      std::accumulate(input_shapes[i].begin(), input_shapes[i].end(), 1,
                      std::multiplies<int64_t>());
  auto buffer_in = std::shared_ptr<float>(new float[input_size],
                                          std::default_delete<float[]>());
  // 读取输入数据
  // ...

  inputs[input_names[i]] = mace::MaceTensor(input_shapes[i], buffer_in);
}

for (size_t i = 0; i < output_count; ++i) {
  int64_t output_size =
      std::accumulate(output_shapes[i].begin(), output_shapes[i].end(), 1,
                      std::multiplies<int64_t>());
  auto buffer_out = std::shared_ptr<float>(new float[output_size],
                                           std::default_delete<float[]>());
  outputs[output_names[i]] = mace::MaceTensor(output_shapes[i], buffer_out);
}

// 5. 执行模型
MaceStatus status = engine.Run(inputs, &outputs);
 ```


 # NCNN

```

 |- lib/libncnn.a
 |- include/ncnn/allocator.h
 |- include/ncnn/blob.h
 |- include/ncnn/command.h
 |- include/ncnn/cpu.h
 |- include/ncnn/datareader.h
 |- include/ncnn/gpu.h
 |- include/ncnn/layer.h
 |- include/ncnn/layer_shader_type.h
 |- include/ncnn/layer_type.h
 |- include/ncnn/mat.h
 |- include/ncnn/modelbin.h
 |- include/ncnn/net.h
 |- include/ncnn/opencv.h
 |- include/ncnn/option.h
 |- include/ncnn/paramdict.h
 |- include/ncnn/pipeline.h
 |- include/ncnn/benchmark.h
 |- include/ncnn/layer_shader_type_enum.h
 |- include/ncnn/layer_type_enum.h
 |- include/ncnn/platform.h
 |- lib/cmake/ncnn/ncnn.cmake
 |- lib/cmake/ncnn/ncnn-release.cmake
 |- lib/cmake/ncnn/ncnnConfig.cmake

./tools/caffe/caffe2ncnn
./tools/mxnet/mxnet2ncnn
./tools/onnx/onnx2ncnn
./tools/darknet/darknet2ncnn
```
 ```c++
    ncnn::Net yolov3;
 
    yolov3.load_param("mobilenetv2_yolov3.param");
    yolov3.load_model("mobilenetv2_yolov3.bin");

    const int target_size = 352;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {0.007843f, 0.007843f, 0.007843f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolov3.create_extractor();
    ex.set_num_threads(4);

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("detection_out", out);
 ```


# Paddle-lite

## 模型压缩

[原理介绍](https://github.com/PaddlePaddle/models/blob/v1.5/PaddleSlim/docs/tutorial.md#4-%E8%BD%BB%E9%87%8F%E7%BA%A7%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2)

## 模型优化工具 opt  
Paddle-Lite 提供了多种策略来自动优化原始的训练模型，其中包括量化、子图融合、混合调度、Kernel优选等等方法。为了使优化过程更加方便易用，我们提供了opt 工具来自动完成优化步骤，输出一个轻量的、最优的可执行模型

## 模型转换工具 X2Paddle  
X2Paddle可以将caffe、tensorflow、onnx模型转换成Paddle支持的模型。

## demo code structure
```c++ 

#include "paddle_api.h"
using namespace paddle::lite_api;

// 1. Set MobileConfig, model_file_path is 
// the path to model model file. 
MobileConfig config;
config.set_model_from_file(model_file_path);
// 2. Create PaddlePredictor by MobileConfig
std::shared_ptr<PaddlePredictor> predictor =
    CreatePaddlePredictor<MobileConfig>(config); 

std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
input_tensor->Resize({1, 3, 224, 224});
auto* data = input_tensor->mutable_data<float>();
for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
  data[i] = 1;
} 

predictor->Run(); 

std::unique_ptr<const Tensor> output_tensor(
    std::move(predictor->GetOutput(0))); 
auto output_data=output_tensor->data<float>();
```