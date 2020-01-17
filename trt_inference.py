# Copyright 2020, Visual Computing Group at HTW. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import abc
import time
import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
from pycuda.autoinit import context as cuda_context


class MemoryBinding(object):
    """
    This class is just a data class
    """
    def __init__(self, name, shape, dtype, host_mem, device_mem, is_input):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.host = host_mem
        self.device = device_mem
        self.binding_id = int(device_mem)
        self.is_input = is_input

    def __str__(self):
        return "Binding of " + self.name + "\nShape:" + str(self.shape) + "\nHost:" + \
               str(self.host) + "\nDevice:" + str(self.device) + "\nIsInput:" + str(self.is_input)

    def __repr__(self):
        return self.__str__()

    def copyTo(self, data):
        """
        Copy data to separated host memory

        :param data: np.ndarray
        """

        if data.shape == self.shape:
            np.copyto(self.host, data.ravel())
        else:
            raise Exception("Shape of array("+str(data.shape)+") is not equal to the shape of input("+str(data.shape)+")")


class TRTInference(object):
    """

    Similar to
    https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/ModelImporter.cpp#L457
    https://github.com/onnx/onnx-tensorrt/blob/ba53ee59da21af3096e38721327c74ec689f0f07/ModelImporter.cpp#L455

    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, serialized_engine, trt_logger_severity=trt.Logger.INFO):
        """
        :param trt_logger_severity:
        """
        start = time.time()

        # make sure the right cuda context is used for the next instructions
        cuda_context.push()

        self._TRT_LOGGER = trt.Logger(trt_logger_severity)
        trt.init_libnvinfer_plugins(self._TRT_LOGGER, "")
        print('Plugin after: {:.0f} [msec]'.format((time.time() - start) * 1000))

        self._runtime = trt.Runtime(self._TRT_LOGGER)
        print('Runtime after: {:.0f} [msec]'.format((time.time() - start) * 1000))

        self._trt_engine = self._runtime.deserialize_cuda_engine(serialized_engine)
        print('Deserialize Cuda engine after: {:.0f} [msec]'.format((time.time() - start) * 1000))

        self._context = self._trt_engine.create_execution_context()
        print('Create execution context after: {:.0f} [msec]'.format((time.time() - start) * 1000))

        self._setup_bindings(self._trt_engine)
        print('Setup bindings after: {:.0f} [msec]'.format((time.time() - start) * 1000))

        # pop the context from the top of the context stack
        cuda_context.pop()

        print("All shapes are known", self._context.all_shape_inputs_specified)
        print("All dynamic shapes are known", self._context.all_binding_shapes_specified)

    def __enter__(self):
        return self

    def __del__(self):
        self._delete_bindings()
        self._context.__del__()
        self._trt_engine.__del__()
        self._runtime.__del__()
        cuda_context.detach()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__del__()
        return True

    def _delete_bindings(self):

        for mem in self._inputs.values():
            mem.device.free()

        for mem in self._outputs:
            mem.device.free()

    def _setup_bindings(self, engine):
        """

        :param engine:
        """

        self._inputs = {}
        self._outputs = []
        self._stream = cuda.Stream()
        for binding in engine:
            name = binding
            shape = engine.get_binding_shape(binding)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            size = trt.volume(shape) * engine.max_batch_size

            # Allocate host and device buffers
            # https://documen.tician.de/pycuda/util.html
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self._inputs[name] = MemoryBinding(name, shape, dtype, host_mem, device_mem, True)
            else:
                self._outputs.append(MemoryBinding(name, shape, dtype, host_mem, device_mem, False))

    def get_input_bindings(self):
        return self._inputs

    def get_output_bindings(self):
        return self._outputs

    def run(self, feed_dict, batch_size=1):
        """

        :param feed_dict: dict<string, np.ndarray>
        :param batch_size:
        :return:
        """

        cuda_context.push()

        bindings = []
        for input_name, np_array in feed_dict.items():
            mem = self._inputs[input_name]
            mem.copyTo(np_array)

            # Transfer input data to the GPU.
            cuda.memcpy_htod(mem.device, mem.host)
            bindings.append(mem.binding_id)

        for mem in self._outputs:
            bindings.append(mem.binding_id)

        if self._context.execute(batch_size=batch_size, bindings=bindings) is False:
            raise Exception("Execution of inference failed, please check the console for error logs.")

        output_dict = {}
        for mem in self._outputs:

            # Transfer predictions back from the GPU.
            cuda.memcpy_dtoh(mem.host, mem.device)

            # Copy to numpy
            output = np.copy(mem.host)
            output = output.reshape(mem.shape)
            output_dict[mem.name] = output

        cuda_context.pop()

        return output_dict
