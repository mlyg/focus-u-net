
# Focus Gate
def FocusGate(input, skip_connections, n_filters, gamma=True, stats='mean', conv_transpose=False):
    """
    Parameters
    ----------
    input : gating signal
    skip_connections : incoming skip connection to be refined
    n_filters : number of filters 
    gamma : whether to apply non-linear attention. Gamma is automatically tuned - see our latest paper
            for more details (https://ieeexplore.ieee.org/document/9761414)
    stats : either 'mean', 'max' or 'mean_max' to apply either global average pooling, global max pooling,
            or both
    conv_transpose : if false applies upsampling 
    """

    # Resize input to match number of channels for skip connections
    resized_input = Conv2D(skip_connections.shape[-1], kernel_size=1, strides=(1, 1), padding='same', use_bias=True)(input)
    # Resize skip connections to match image shape for input
    resized_skip = Conv2D(skip_connections.shape[-1], kernel_size=1, strides=(2, 2), padding='same', use_bias=False)(skip_connections)

    stride_x = resized_skip.shape[1] // input.shape[1]
    stride_y = resized_skip.shape[2] // input.shape[2]

    if conv_transpose:
        resized_input = Conv2DTranspose(skip_connections.shape[-1], (stride_x, stride_y),strides=(stride_x, stride_y),padding='same')(resized_input)
    else:
        resized_input = UpSampling2D((stride_x,stride_y))(resized_input)

    # element wise addition 
    added  = add([resized_input, resized_skip])

    # perform non-linear activation
    act = Activation('relu')(added)

    # channel attention
    channel_attention = channel_attention_block(act, stats)

    # spatial attention
    spatial_attention = spatial_attention_block(act, stats)

    # combine channel and spatial weights
    weights = multiply([channel_attention, spatial_attention])

    # focal parameter for tuneable background suppression
    if gamma:
        weights = Gamma()(weights)

    # rescale attention coefficient matrix to size of skip connection
    stride_x_weights = skip_connections.shape[1] // weights.shape[1]
    stride_y_weights = skip_connections.shape[2] // weights.shape[2]
    
    # Upsample to match original skip connection resolution
    if conv_transpose:
        weights = Conv2DTranspose(skip_connections.shape[-1], (stride_x_weights, stride_y_weights), strides=(stride_x_weights, stride_y_weights), padding='same')(weights)
    else:
        weights = UpSampling2D((stride_x_weights, stride_y_weights))(weights) 

    # multiply skip connections by weights
    weights = multiply([weights, skip_connections])

    # perform final convolution and batch normalisation
    output = Conv2D(skip_connections.shape[-1], kernel_size=1, strides=(1, 1), padding='same', use_bias=True)(weights)
    output = BatchNormalization()(output)

    return output

# Automatically tuned gamma parameter
class Gamma(tf.keras.layers.Layer):

    def __init__(self):
        super(Gamma, self).__init__()

    def build(self, input_shape):
        initializer = tf.keras.initializers.Ones()

        self.w = self.add_weight(
            shape=(1,1),
            initializer=initializer,
            trainable=True,
            constraint= tf.keras.constraints.NonNeg())

    def call(self, inputs):
        inputs = tf.clip_by_value(inputs, tf.keras.backend.epsilon(), 1.)
        return (inputs)**self.w


# Gating signal
def GatingSignal(input, skip_connections):
    """
    Parameters
    ----------
    input : layer prior to upsampling
    skip_connections : incoming skip connection to be refined
    """
    signal = Conv2D(skip_connections.shape[-1], (1, 1), strides=(1, 1), padding="same", activation='relu')(input)
    signal = BatchNormalization()(signal)
    return signal

def spatial_kernel_size(input, beta=1, gamma=2):
    """
    The original implementation based the kernel size on the channel, but with
    square images, it can be simplified to:
    k = | log2(R)/gamma + b/gamma|odd
    """

    k = int(abs(np.log2((input.shape[1])/gamma) + beta/gamma))
    out = k if k % 2 else k + 1

    return out

def channel_kernel_size(input, beta=1,gamma=2):
    """
    k = | log2(C)/gamma + b/gamma|odd
    """
    k = int(abs(np.log2(input.shape[-1]/gamma) + beta/gamma))
    out = k if k % 2 else k + 1
    return out

def spatial_attention_block(input, stats):
    """ Adaptive spatial attention block
    """

    k_size = spatial_kernel_size(input)

    # generate spatial statistics
    if (stats == 'mean') or (stats == 'mean_max'):
        mean = tf.reduce_mean(input, axis=-1, keepdims=True)
        t = mean

    if (stats == 'max') or (stats == 'mean_max'):
        maximum = tf.reduce_max(input, axis=-1, keepdims=True)
        t = maximum

    if (stats == 'mean_max'):
        t = concatenate([mean,maximum], axis=-1)

    t = Conv2D(1, kernel_size=k_size, padding="same",activation='sigmoid', use_bias = False)(t)

    return t

def channel_attention_block(input, stats):
    """ Adaptive channel attention block
    """

    k_size = channel_kernel_size(input)

    # generate channel statistics
    if (stats == 'mean') or (stats == 'mean_max'):
        mean = tf.reduce_mean(input, axis=[1,2], keepdims=True)
        mean = tf.squeeze(mean,axis=(-2))
        mean = tf.transpose(mean,perm=[0,2,1])
        t = mean

    if (stats == 'max') or (stats == 'mean_max'):
        maximum = tf.reduce_max(input, axis=[1,2], keepdims=True)
        maximum = tf.squeeze(maximum,axis=(-2))
        maximum = tf.transpose(maximum,perm=[0,2,1])
        t = maximum

    if (stats == 'mean_max'):
        t = concatenate([mean,maximum], axis=-1)

    t = Conv1D(filters=1, kernel_size=k_size, padding='same', use_bias=False)(t)
    t = tf.transpose(t,perm=[0,2,1])
    t = tf.expand_dims(t,(-2))
    t = tf.math.sigmoid(t)

    return t