import tensorflow as tf
import numpy as np

arr_hr = 0
arr_lr = 0
work_dir = ''

parms = {'verbose': 0, 'callbacks': [TQDMNotebookCallback(leave_inner=True)]

def conv_block(x, filters, size, stride=(2,2), mode='same', act=True):
    x = tf.keras.layers.Conv2D(filters, size, size, subsample=stride, border_mode=mode)(x)
    x = tf.keras.layers.BatchNormalization(mode=2)(x)
    return tf.keras.layers.Activation('relu')(x) if act else x

def res_block(ip, nf=64):
    x = conv_block(ip, nf, 3, (1,1))
    x = conv_block(x, nf, 3, (1,1), act=False)
    return merge([x, ip], mode='sum')

def deconv_block(x, filters, size, shape, stride=(2,2)):
    x = tf.keras.layers.Deconvolution2D(filters, size, size, subsample=stride,
        border_mode='same', output_shape=(None,)+shape)(x)
    x = tf.keras.layers.BatchNormalization(mode=2)(x)
    return tf.keras.layers.Activation('relu')(x)

def up_block(x, filters, size):
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(filters, size, size, border_mode='same')(x)
    x = tf.keras.layers.BatchNormalization(mode=2)(x)
    return tf.keras.layers.Activation('relu')(x)


inp = tf.keras.layers.Input(arr_lr.shape[1:])
x=conv_block(inp, 64, 9, (1,1))

for i in range(4):
    x=res_block(x)

x = up_block(x, 64, 3)
x = up_block(x, 64, 3)
x = tf.keras.layers.Conv2D(3, 9, 9, activation='tanh', border_mode='same')(x)
outp = tf.keras.layers.Lambda(lambda x: (x+1)*127.5)(x)

vgg_inp = tf.keras.layers.Input(shp)
vgg = VGG16(include_top=False, input_tensor=layers.Lambda(preprocess)(vgg_inp))

for l in vgg.layers: l.trainable=False

def get_outp(m, ln):
    return m.get_layer(f'block{ln}_conv1').output

vgg_content = tf.keras.models.Model(vgg_inp, [get_outp(vgg, o) for o in [1,2,3]])
vgg1 = vgg_content(vgg_inp)
vgg2 = vgg_content(outp)


def mean_sqr_b(diff):
    dims = list(range(1,tf.ndim(diff)))
    return tf.expand_dims(tf.sqrt(tf.mean(diff**2, dims)), 0)


w=[0.1, 0.8, 0.1]

def content_fn(x):
    res = 0; n=len(w)
    for i in range(n):
        res += mean_sqr_b(x[i]-x[i+n]) * w[i]
    return res


m_sr = tf.keras.models.Model([inp, vgg_inp], tf.keras.layers.Lambda(content_fn)(vgg1+vgg2))
targ = np.zeros((arr_hr.shape[0], 1))

m_sr.compile('adam', 'mse')
m_sr.fit([arr_lr, arr_hr], targ, 8, 2, **parms)

tf.set_value(m_sr.optimizer.lr, 1e-4)
m_sr.fit([arr_lr, arr_hr], targ, 16, 1, **parms)


top_model = tf.keras.models.Model(inp, outp)

p = top_model.predict(arr_lr[10:11])

top_model.save_weights(work_dir + 'sr_final.h5')
top_model.load_weights(work_dir + 'top_final.h5')