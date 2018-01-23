from grpc.beta import implementations
import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


concurrency = 1
num_tests = 100
host = ''
port = 8000
work_dir = '/tmp'



def _create_rpc_callback():
  def _callback(result):
      response = numpy.array(
        result.result().outputs['y'].float_val)
      prediction = numpy.argmax(response)
      print(prediction)
  return _callback


test_data_set = mnist.test
test_image = mnist.test.images[0]

predict_request = predict_pb2.PredictRequest()
predict_request.model_spec.name = 'mnist'
predict_request.model_spec.signature_name = 'prediction'

predict_channel = implementations.insecure_channel(host, int(port))
predict_stub = prediction_service_pb2.beta_create_PredictionService_stub(predict_channel)

predict_request.inputs['x'].CopyFrom(
    tf.contrib.util.make_tensor_proto(test_image, shape=[1, test_image.size]))
result = predict_stub.Predict.future(predict_request, 3.0)
result.add_done_callback(
    _create_rpc_callback())
