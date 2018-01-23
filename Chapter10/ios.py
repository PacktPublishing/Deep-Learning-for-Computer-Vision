import tfcoreml as tf_converter
tf_converter.convert(tf_model_path='tf_model_path.pb',
                     mlmodel_path='mlmodel_path.mlmodel',
                     output_feature_names=['softmax:0'],
                     input_name_shape_dict={'input:0': [1, 227, 227, 3]})