# Tensorflow to TensorRT converter (TF2TRT)
This converter parses a [Tensorflow](https://www.tensorflow.org/) protobuf file or graph_def object and creates a [TensorRT](https://developer.nvidia.com/tensorrt) network out of it. Everything is written in Python 3 and does not require the installation of additional packages.

### Contents
 - [Limitations](#limitations)
 - [Requirements](#requirements)
 - [Why use TF2TRT?](#why-use-tf2trt)
 - [Usage](#usage)
 - [Contribution](#contribution)
 - [Credits](#credits)


# Limitations
There is **no support for dynamic shapes**. Computations based on shapes of tensors inside the network at runtime are therefore not possible. Every shape musst be fully specified at construction time in Tensorflow. 

Tensorflow has a lot of custom made operations and not all of them are supported in TensorRT. Right now the converter is **missing a lot of operations and attributes**. Adding them is quite easy if they are supported by TensorRT. 

The layout of the input data into the Tensorflow network should be **channel-first (NCHW)** making the conversion easier.

# Requirements
* [pycuda 2019.1.2](https://documen.tician.de/pycuda/)
* [numpy 1.16](https://numpy.org/)
* [TensorRT 6.0](https://developer.nvidia.com/tensorrt)
* [Tensorflow 1.14](https://www.tensorflow.org/)

It is recommended to install [TensorRT](https://developer.nvidia.com/tensorrt) via [Anaconda](https://www.anaconda.com/) and the [IBM repositories](https://www.ibm.com/support/knowledgecenter/SS5SF7_1.6.2/navigation/wmlce_install.html). The newest pycuda version can be installed with [pypi](https://pypi.org/project/pycuda/) or [IBMs channel](https://anaconda.org/powerai/pycuda) in Anaconda.

# Why use TF2TRT?
* Easy to extend
* No special 3rd party dependencies
* Pure Python

If you are interested in a full fledged converter try out [ONNX-TensorRT](https://github.com/onnx/onnx-tensorrt/)

# Usage
The shortest way to use the converter is just to call 
```python
import tensorrt as trt
from trt_importer import TRTImporter

importer = TRTImporter(trt.Logger.VERBOSE)
network = importer.from_tensorflow_graph_def(graph_def, ["input_tensor"], [[1, 3, 224, 224]], ["softmax"])
serialized_engine = importer.optimize_network(network, max_workspace_size=4 * (1 << 30))
```

The [example](https://github.com/Visual-Computing/TF2TRT/blob/master/TF%20to%20TRT%20converter%20example.ipynb) also explains how to froze a Tensorflow graph and run a TensorRT engine.

### How to use FP16?
The TensorRT network can be optimized for FP16 computation by adding a ```fp16_mode=True``` parameter to the ```optimize_network(...)``` method. Please note it is important to still provide a normal FP32 graph and do not have any manual FP16 casting operation in it. 

### Optimize for bigger batch sizes
An additional parameter ```max_batch_size=32``` of the ```optimize_network(...)``` method will define the maximal batch size the resulting engine will be able to execute. Optimizing for higher batch sizes might reduce the performance of smaller ones. If you need perfect performance for several batch sizes, you need to create execution profiles for each one of them.

### Quality vs speed up the optimization process
Since the optimization process is an exaustic search of implementations to run a particular layer in the least amount of time. We can specify how often each implementation should be repreated (```min_find_iterations```, ```average_find_iterations```) in order to get an average computation time. Our API provides a simplified parameter for that ```importer.optimize_network(..., fast_pass=True)``` which sets both iteration parameter to 1 if Fast Pass is True and otherwise to 5.

### Expected memory consumption?
The resulting serialized engine contains the weights and structure of the network. When executing an inference pass all the weights will be copied into the memory of the GPU plus some additional space for intermediate calculations. The ```serialized_engine.device_memory_size``` variables gives an approximation of how many memory will be consumed. 

# Contribution
We love to get in contact with the community. Feel free to [e-mail](mailto:info@visual-computing.com) us or use the [issue system](https://github.com/Visual-Computing/TF2TRT/issues) to suggest new features and ask questions. Pull requests are always welcome, we try to incorporate them into the master branch as fast as possible. Not sure if that typo is worth a pull request? Do it! We will appreciate it.

# Credits
This project is maintained by the [Visual Computing Group at HTW Berlin](https://visual-computing.com/aboutus/). Some parts of the source code are based on methods of the [ONNX-TensorRT](https://github.com/onnx/onnx-tensorrt/tree/6.0-full-dims/) converter.




