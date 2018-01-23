import tensorflow as tf
from keras.layers.recurrent import *

training = True
sequence_length = 0
vocabulary_size = 0
input_tensor = 0
input_shape = 0
embedding_dimension = 0
dropout_prob = 0.3
previous_words = 0
height = 0
shape = 0
cnn_features = 0
depth = 0

vgg_model = tf.keras.applications.vgg16.VGG16(weights='imagenet',
                                              include_top=False,
                                              input_tensor=input_tensor,
                                              input_shape=input_shape)

word_embedding = tf.keras.layers.Embedding(
    vocabulary_size, embedding_dimension, input_length=sequence_length)
embbedding = word_embedding(previous_words)
embbedding = tf.keras.layers.Activation('relu')(embbedding)
embbedding = tf.keras.layers.Dropout(dropout_prob)(embbedding)

cnn_features_flattened = tf.keras.layers.Reshape((height * height, shape))(cnn_features)
net = tf.keras.layers.GlobalAveragePooling1D()(cnn_features_flattened)

net = tf.keras.layers.Dense(embedding_dimension, activation='relu')(net)
net = tf.keras.layers.Dropout(dropout_prob)(net)
net = tf.keras.layers.RepeatVector(sequence_length)(net)
net = tf.keras.layers.concatenate()([net, embbedding])
net = tf.keras.layers.Dropout(dropout_prob)(net)




class LSTM_sent(Recurrent):
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., sentinel=True, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U
        self.sentinel = sentinel
        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(LSTM_sent, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        if self.stateful:
            self.reset_states()
        else:
            if self.sentinel:
                self.states = [None, None]
            else:
                self.states = [None]

        self.W_i = self.init((input_dim, self.output_dim),
                             name='{}_W_i'.format(self.name))
        self.U_i = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_i'.format(self.name))
        self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))

        self.W_f = self.init((input_dim, self.output_dim),
                             name='{}_W_f'.format(self.name))
        self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_f'.format(self.name))
        self.b_f = self.forget_bias_init((self.output_dim,),
                                         name='{}_b_f'.format(self.name))

        self.W_c = self.init((input_dim, self.output_dim),
                             name='{}_W_c'.format(self.name))
        self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_c'.format(self.name))
        self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))

        self.W_o = self.init((input_dim, self.output_dim),
                             name='{}_W_o'.format(self.name))
        self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_o'.format(self.name))
        self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))

        if self.sentinel:
            # sentinel gate
            self.W_g = self.init((input_dim, self.output_dim),
                                 name='{}_W_g'.format(self.name))
            self.U_g = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_g'.format(self.name))
            self.b_g = K.zeros((self.output_dim,), name='{}_b_g'.format(self.name))


            self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                      self.W_c, self.U_c, self.b_c,
                                      self.W_f, self.U_f, self.b_f,
                                      self.W_o, self.U_o, self.b_o,
                                      self.W_g, self.U_g, self.b_g]
        else:
            self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                      self.W_c, self.U_c, self.b_c,
                                      self.W_f, self.U_f, self.b_f,
                                      self.W_o, self.U_o, self.b_o]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x, train=False):
        if self.consume_less == 'cpu':
            if train and (0 < self.dropout_W < 1):
                dropout = self.dropout_W
            else:
                dropout = 0
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x_i = time_distributed_dense(x, self.W_i, self.b_i, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_f = time_distributed_dense(x, self.W_f, self.b_f, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_c = time_distributed_dense(x, self.W_c, self.b_c, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_o = time_distributed_dense(x, self.W_o, self.b_o, dropout,
                                         input_dim, self.output_dim, timesteps)
            if self.sentinel:
                x_g = time_distributed_dense(x, self.W_g, self.b_g, dropout,
                                             input_dim, self.output_dim, timesteps)
                return K.concatenate([x_i, x_f, x_c, x_o,x_g], axis=2)
            else:
                return K.concatenate([x_i, x_f, x_c, x_o], axis=2)

        else:
            return x

    def step(self, x, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        B_U = states[2]
        B_W = states[3]

        if self.consume_less == 'cpu':
            x_i = x[:, :self.output_dim]
            x_f = x[:, self.output_dim: 2 * self.output_dim]
            x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
            x_o = x[:, 3 * self.output_dim: 4 * self.output_dim]
            if self.sentinel:
                x_g = x[:, 4 * self.output_dim:]
        else:
            x_i = K.dot(x, self.W_i) + self.b_i
            x_f = K.dot(x * B_W[1], self.W_f) + self.b_f
            x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
            x_o = K.dot(x * B_W[3], self.W_o) + self.b_o
            if self.sentinel:
                x_g = K.dot(x * B_W[4], self.W_g) + self.b_g

        i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
        f = self.inner_activation(x_f + K.dot(h_tm1 * B_U[1], self.U_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
        o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[3], self.U_o))
        h = o * self.activation(c)
        if self.sentinel:
            g = self.inner_activation(x_g + K.dot(h_tm1 * B_U[4], self.U_g))
            s = g * self.activation(c)
            return [h,s], [h, c]
        else:
            return h, [h, c]

    def get_constants(self, x):
        constants = []
        if self.sentinel:
            Ngate = 5
        else:
            Ngate = 4
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.dropout(ones, self.dropout_U) for _ in range(Ngate)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(Ngate)])

        if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.dropout(ones, self.dropout_W) for _ in range(Ngate)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(Ngate)])
        return constants


    def get_output_shape_for(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) > 1:
            input_shape = input_shape[0]
        if self.return_sequences:
            output_shape = (input_shape[0], input_shape[1], self.output_dim)
        else:
            output_shape = (input_shape[0], self.output_dim)
        #the hidden state and the sentinel have the same shape
        if self.sentinel:
            return [output_shape, output_shape]
        else:
            return output_shape

    def compute_mask(self, input, mask):
        if self.return_sequences:
            if self.sentinel:
                return [mask, mask]
            else :
                return mask
        else:
            if self.sentinel:
                return [None, None]
            else:
                return None

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])


        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))
        if self.sentinel:
            outputs = K.permute_dimensions(outputs, [0,2,1,3])
            if self.return_sequences:
                return [outputs[0], outputs[1]]
            else:
                return [last_output[0],last_output[1]]
        else:
            if self.return_sequences:
                return outputs
            else:
                return last_output


    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U,
                  "sentinel": self.sentinel}
        base_config = super(LSTM_sent, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




lstm_ = LSTM_sent(output_dim = args_dict.lstm_dim,
                  return_sequences=True,stateful=True,
                  dropout_W = dropout_prob,
                  dropout_U = dropout_prob,
                  sentinel=True,name='hs')
h, s = lstm_(x)

num_vfeats = wh * wh
num_vfeats = num_vfeats + 1


h_out_linear = tf.keras.layers.Convolution1D(
    depth, 1, activation='tanh', border_mode='same')(h)
h_out_linear = tf.keras.layers.Dropout(
    dropout_prob)(h_out_linear)
h_out_embed = tf.keras.layers.Convolution1D(
    embedding_dimension, 1, border_mode='same')(h_out_linear)
z_h_embed = tf.keras.layers.TimeDistributed(
    tf.keras.layers.RepeatVector(num_vfeats))(h_out_embed)

Vi = tf.keras.layers.Convolution1D(
    depth, 1, border_mode='same', activation='relu')(V)

Vi = tf.keras.layers.Dropout(dropout_prob)(Vi)
Vi_emb = tf.keras.layers.Convolution1D(
    embedding_dimension, 1, border_mode='same', activation='relu')(Vi)

z_v_linear = tf.keras.layers.TimeDistributed(
    tf.keras.layers.RepeatVector(sequence_length))(Vi)
z_v_embed = tf.keras.layers.TimeDistributed(
    tf.keras.layers.RepeatVector(sequence_length))(Vi_emb)

z_v_linear = tf.keras.layers.Permute((2, 1, 3))(z_v_linear)
z_v_embed = tf.keras.layers.Permute((2, 1, 3))(z_v_embed)

fake_feat = tf.keras.layers.Convolution1D(
    depth, 1, activation='relu', border_mode='same')(s)
fake_feat = tf.keras.layers.Dropout(dropout_prob)(fake_feat)

fake_feat_embed = tf.keras.layers.Convolution1D(
    embedding_dimension, 1, border_mode='same')(fake_feat)
z_s_linear = tf.keras.layers.Reshape((sequence_length, 1, depth))(fake_feat)
z_s_embed = tf.keras.layers.Reshape(
    (sequence_length, 1, embedding_dimension))(fake_feat_embed)

z_v_linear = tf.keras.layers.concatenate(axis=-2)([z_v_linear, z_s_linear])
z_v_embed = tf.keras.layers.concatenate(axis=-2)([z_v_embed, z_s_embed])

z = tf.keras.layers.Merge(mode='sum')([z_h_embed,z_v_embed])
z = tf.keras.layers.Dropout(dropout_prob)(z)
z = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Activation('tanh'))(z)
attention= tf.keras.layers.TimeDistributed(
    tf.keras.layers.Convolution1D(1, 1, border_mode='same'))(z)

attention = tf.keras.layers.Reshape((sequence_length, num_vfeats))(attention)
attention = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Activation('softmax'))(attention)
attention = tf.keras.layers.TimeDistributed(
    tf.keras.layers.RepeatVector(depth))(attention)
attention = tf.keras.layers.Permute((1,3,2))(attention)
w_Vi = tf.keras.layers.Add()([attention,z_v_linear])
sumpool = tf.keras.layers.Lambda(lambda x: K.sum(x, axis=-2),
               output_shape=(depth,))
c_vec = tf.keras.layers.TimeDistributed(sumpool)(w_Vi)
atten_out = tf.keras.layers.Merge(mode='sum')([h_out_linear,c_vec])
h = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Dense(embedding_dimension,activation='tanh'))(atten_out)
h = tf.keras.layers.Dropout(dropout_prob)(h)

predictions = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Dense(vocabulary_size, activation='softmax'))(h)

model = tf.keras.models.Model(input=[cnn_features, prev_words], output=predictions)
opt = get_opt(args_dict)


in_im = tf.keras.layers.Input(
    batch_shape=(args_dict.bs, args_dict.imsize, args_dict.imsize, 3), name='image')

wh = vgg_model.output_shape[1]
dim = vgg_model.output_shape[3]

if not args_dict.cnn_train:
    for i,layer in enumerate(convnet.layers):
        if i > args_dict.finetune_start_layer:
            layer.trainable = False

imfeats = vgg_model(in_im)
cnn_features = tf.keras.layers.Input(batch_shape=(args_dict.bs, wh, wh, dim))
prev_words = tf.keras.layers.Input(batch_shape=(args_dict.bs, sequence_length))
lang_model = language_model(args_dict, wh, dim, cnn_features, prev_words)

out = lang_model([imfeats,prev_words])

model = tf.keras.models.Model(input=[in_im, prev_words], output=out)
