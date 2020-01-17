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
import tensorrt as trt
import tensorflow as tf

from viscoop.tensorrt.trt_builder import TRTNetworkBuilder
from viscoop.tensorrt.tf_parser import TFProtobufParser
from viscoop.tensorrt.trt_inference import TRTInference

from tensorflow.python.platform import gfile


class TRTImporter(object):
    """

    Similar to
    https://github.com/onnx/onnx-tensorrt/blob/6.0-full-dims/ModelImporter.cpp#L457
    https://github.com/onnx/onnx-tensorrt/blob/ba53ee59da21af3096e38721327c74ec689f0f07/ModelImporter.cpp#L455

    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, trt_logger_severity=trt.Logger.INFO):
        """

        :param trt_logger_severity:
        """

        self._trt_logger_severity = trt_logger_severity
        self._TRT_LOGGER = trt.Logger(trt_logger_severity)
        trt.init_libnvinfer_plugins(self._TRT_LOGGER, "")

    def from_tensorflow_pb_file(self, pb_file, input_node_names, input_node_shapes, output_node_names):
        """
        Convert the graph in the pb file to a trt network

        :param pb_file:
        :param input_node_names:
        :param input_node_shapes:
        :param output_node_names:
        :return: TRT Network
        """

        # read pb file
        graph_def = tf.GraphDef()
        with gfile.FastGFile(pb_file, 'rb') as f:
            graph_def.ParseFromString(f.read())
        return self.from_tensorflow_graph_def(graph_def, input_node_names, input_node_shapes, output_node_names)

    def from_tensorflow_graph_def(self, frozen_graph_def, input_node_names, input_node_shapes, output_node_names):

        # setup TRT network builder
        with trt.Builder(self._TRT_LOGGER) as builder:
            network = builder.create_network()
        network_builder = TRTNetworkBuilder(network)
        network_builder.register_inputs(input_node_names, input_node_shapes)
        network_builder.register_outputs(output_node_names)

        # parse Tensorflow PB file and build TRT network
        parser = TFProtobufParser(frozen_graph_def, network_builder)
        if parser.parse():
            return network_builder.network

        # Something went wrong
        return None

    def optimize_network(self, network, max_batch_size=1, fp16_mode=False, max_workspace_size=4294967296, fast_pass=False):
        """
        Optimize a TRT network and create a TRT engine out of it

        :param network:
        :param max_batch_size:
        :param fp16_mode:
        :param max_workspace_size:
        :param fast_pass:
        :return: TRT Engine
        """

        with trt.Builder(self._TRT_LOGGER) as builder:
            builder.max_batch_size = max_batch_size
            builder.max_workspace_size = max_workspace_size
            builder.fp16_mode = fp16_mode
            builder.min_find_iterations = 1 if fast_pass else 5
            builder.average_find_iterations = 1 if fast_pass else 5
            print("max_workspace_size", builder.max_workspace_size)
            print("average_find_iterations", builder.average_find_iterations)
            print("min_find_iterations", builder.min_find_iterations)

            # build and serialize the network
            st = time.time()
            engine = builder.build_cuda_engine(network)
            print('finished building an engine in {0:.2f} [sec]'.format((time.time() - st)))
            print("num_layers", engine.num_layers)
            print("device_memory_size", engine.device_memory_size)
            return engine.serialize()

    def inference_engine(self, engine):
        """
        Setup and return an inference context

        :param engine:
        :return: inference context
        """
        return TRTInference(engine, self._trt_logger_severity)

    @staticmethod
    def store_engine(engine, engine_file):
        """
        Store a TRT engine to the hard drive

        :param engine:
        :param engine_file:
        :return: bool
        """

        with open(engine_file, "wb") as f:
            num_chars = f.write(engine)
        print("stored TensorRT engine", engine_file)
        return num_chars > 1

    @staticmethod
    def load_engine(engine_file):
        """
        Load a TRT engine from the hard drive

        :param engine_file:
        :return: TRT Engine
        """

        # read from file
        with open(engine_file, "rb") as f:
            return f.read()