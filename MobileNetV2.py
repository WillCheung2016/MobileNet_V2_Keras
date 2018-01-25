from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import AveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Conv2D
from keras.layers import add
from keras.layers import Dense
from keras.layers import Flatten
from keras import initializers
from keras.regularizers import l2
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras import backend as K
from keras.applications.mobilenet import relu6, DepthwiseConv2D
import tensorflow as tf

def Relu6(x, **kwargs):
    return Activation(relu6, **kwargs)(x)

def InvertedResidualBlock(x, expand, out_channels, repeats, stride, weight_decay, block_id):

    in_channels = K.int_shape(x)[-1]
    x = Conv2D(expand * in_channels, 1, padding='same', strides=stride, use_bias=False,
                kernel_regularizer=l2(weight_decay), name='conv_%d_0' % block_id)(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.9, name='conv_%d_0_bn' % block_id)(x)
    x = Relu6(x, name='conv_%d_0_act_1' % block_id)
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=1,
                        strides=1,
                        use_bias=False,
                        kernel_regularizer=l2(weight_decay),
                        name='conv_dw_%d_0' % block_id )(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.9, name='conv_dw_%d_0_bn' % block_id)(x)
    x = Relu6(x, name='conv_%d_0_act_2' % block_id)
    x = Conv2D(out_channels, 1, padding='same', strides=1, use_bias=False,
               kernel_regularizer=l2(weight_decay), name='conv_bottleneck_%d_0' % block_id)(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.9, name='conv_bottlenet_%d_0_bn' % block_id)(x)

    for i in xrange(1, repeats):
        x1 = Conv2D(expand*out_channels, 1, padding='same', strides=1, use_bias=False,
                    kernel_regularizer=l2(weight_decay), name='conv_%d_%d' % (block_id, i))(x)
        x1 = BatchNormalization(epsilon=1e-5,momentum=0.9,name='conv_%d_%d_bn' % (block_id, i))(x1)
        x1 = Relu6(x1,name='conv_%d_%d_act_1' % (block_id, i))
        x1 = DepthwiseConv2D((3, 3),
                            padding='same',
                            depth_multiplier=1,
                            strides=1,
                            use_bias=False,
                            kernel_regularizer=l2(weight_decay),
                            name='conv_dw_%d_%d' % (block_id, i))(x1)
        x1 = BatchNormalization(epsilon=1e-5,momentum=0.9, name='conv_dw_%d_%d_bn' % (block_id, i))(x1)
        x1 = Relu6(x1, name='conv_dw_%d_%d_act_2' % (block_id, i))
        x1 = Conv2D(out_channels, 1, padding='same', strides=1, use_bias=False,
                    kernel_regularizer=l2(weight_decay),name='conv_bottleneck_%d_%d' % (block_id, i))(x1)
        x1 = BatchNormalization(epsilon=1e-5, momentum=0.9, name='conv_bottlenet_%d_%d_bn' % (block_id, i))(x1)
        x = add([x, x1], name='block_%d_%d_output' % (block_id, i))
    return x

def MobileNetV2(input_shape, classes, weight_decay,input_tensor=None):
    if input_tensor is not None:
        img_input = Input(tensor=input_tensor)
    else:
        img_input = Input(input_shape)

    x = conv_block(img_input, 32, name='conv1',alpha=1, strides=(2, 2))
    x = InvertedResidualBlock(x, expand=1, out_channels=16, repeats=1, stride=1, weight_decay=weight_decay, block_id=1)
    x = InvertedResidualBlock(x, expand=6, out_channels=24, repeats=2, stride=2, weight_decay=weight_decay, block_id=2)
    x = InvertedResidualBlock(x, expand=6, out_channels=32, repeats=3, stride=2, weight_decay=weight_decay, block_id=3)
    x = InvertedResidualBlock(x, expand=6, out_channels=64, repeats=4, stride=1, weight_decay=weight_decay, block_id=4)
    x = InvertedResidualBlock(x, expand=6, out_channels=96, repeats=3, stride=2, weight_decay=weight_decay, block_id=5)
    x = InvertedResidualBlock(x, expand=6, out_channels=160, repeats=3, stride=2, weight_decay=weight_decay, block_id=6)
    x = InvertedResidualBlock(x, expand=6, out_channels=320, repeats=1, stride=1, weight_decay=weight_decay, block_id=7)
    x = conv_block(x, 1280, name='conv2', alpha=1, kernel=(1, 1),strides=1)
    
    x = AveragePooling2D((7, 7))(x)
    x = Flatten()(x)
    x = Dense(classes, kernel_regularizer=l2(weight_decay), name='fc_pred')(x)
    x = Activation('softmax', name='act_softmax')(x)

    return Model(inputs=img_input, outputs=x)

def conv_block(inputs, filters, alpha, name,kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name=name)(inputs)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-5,momentum=0.9,name=name+'_bn')(x)
    return Relu6(x, name=name+'_relu')
