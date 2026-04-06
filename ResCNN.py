import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_gru_model(ntimes_input, nlev, nfeat, out_vars=4, hidden=128, dropout=0.1):
    inp = layers.Input(shape=(ntimes_input, nlev, nfeat))

    # 每个时次先编码成柱特征
    x = layers.TimeDistributed(layers.Flatten())(inp)
    x = layers.TimeDistributed(layers.Dense(hidden, activation="relu"))(x)

    x = layers.GRU(hidden, return_sequences=False, dropout=dropout)(x)
    x = layers.Dense(nlev * out_vars, activation="linear")(x)
    out = layers.Reshape((nlev, out_vars))(x)
    return Model(inp, out, name="GRU_column")

def build_mlp_model(ntimes_input, nlev, nfeat, out_vars=4, hidden=512, dropout=0.1):
    inp = layers.Input(shape=(ntimes_input, nlev, nfeat))
    x = layers.Flatten()(inp)
    x = layers.Dense(hidden, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(hidden, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(nlev * out_vars, activation="linear")(x)
    out = layers.Reshape((nlev, out_vars))(x)
    return Model(inp, out, name="MLP_baseline")

def residual_block_1d(x, filters, kernel_size=3, dilation_rate=1,
                      dropout_rate=0.1, l2=1e-5, use_bn=True):
    shortcut = x

    y = x
    if use_bn:
        y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        kernel_regularizer=regularizers.l2(l2),
        use_bias=not use_bn
    )(y)

    if use_bn:
        y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Dropout(dropout_rate)(y)
    y = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        kernel_regularizer=regularizers.l2(l2),
        use_bias=not use_bn
    )(y)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)

    out = layers.Add()([shortcut, y])
    return out

def build_rescnn_model(ntimes_input, nlev, nfeat, out_vars=4,
                       filters=256, kernel_size=3, n_blocks=4, dropout=0.1):
    inp = layers.Input(shape=(ntimes_input, nlev, nfeat))

    # 把时间窗和特征并到 channel，沿 lev 做卷积
    x = layers.Permute((2, 1, 3))(inp)              # (lev, time, feat)
    x = layers.Reshape((nlev, ntimes_input * nfeat))(x)

    x = layers.Conv1D(filters, 1, padding="same")(x)

    dilation_list = [1, 2, 4, 8, 16, 32]
    for i in range(n_blocks):
        x = residual_block_1d(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_list[i % len(dilation_list)],
            dropout_rate=dropout,
            use_bn=True
        )
    x_mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    out = layers.Conv1D(out_vars, 1, padding="same", activation="tanh")(x)
    return Model(inp, out, name="ResCNN_TCN")

def build_mlp_model(ntimes_input, nlev, nfeat, out_vars=4, hidden=512, dropout=0.1):
    inp = layers.Input(shape=(ntimes_input, nlev, nfeat))
    x = layers.Flatten()(inp)
    x = layers.Dense(hidden, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(hidden, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(nlev * out_vars, activation="tanh")(x)
    out = layers.Reshape((nlev, out_vars))(x)
    return Model(inp, out, name="MLP_baseline")

def _bn_relu_conv(filters, kernel_size, bn=True):
    def f(inputs):
        x = inputs
        if bn:
            x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        if x.shape[1] is not None and x.shape[1] > 26:
            x = layers.Lambda(lambda z: z[:, 1:, :])(x)

        x = layers.Conv1D(
            filters=filters * 2,
            kernel_size=kernel_size,
            padding="same",
            use_bias=not bn,
        )(x)
        return x
    return f


def _fc(n):
    def f(inputs):
        x = layers.Dense(n, activation="relu")(inputs)
        return x
    return f


def zm_model(ntimes_input, nlev, nfeat, out_vars=4,
                        num_filter=128, kernel_size=3, num_blocks=6, fn_size=16, bn=False):
    """
    把现在输入 (batch, time, lev, feat)
    转成zm_model使用的 (batch, lev, channel)
    """
    inp = layers.Input(shape=(ntimes_input, nlev, nfeat))

    # (batch, time, lev, feat) -> (batch, lev, time*feat)
    x = layers.Permute((2, 1, 3))(inp)
    x = layers.Reshape((nlev, ntimes_input * nfeat))(x)


    outputs = _bn_relu_conv(num_filter, kernel_size, bn=bn)(x)

    for ii in range(num_blocks - 1):
        if ii < 1:
            outputs = _bn_relu_conv(num_filter, kernel_size, bn=bn)(outputs)
        else:
            outputs = _bn_relu_conv(num_filter, kernel_size, bn=bn)(outputs)
        outputs = _fc(fn_size)(outputs)

    outputs = layers.Dense(nlev, activation=None, name="reconstruct_vertical")(outputs)
    outputs = layers.Permute((2, 1))(outputs)
    outputs = layers.Dense(out_vars, activation="tanh")(outputs)

    return Model(inp, outputs, name="zm_model")


def build_2dconv_model(nlev, nfeat, out_vars, ntimes_input=5):
    inp  = layers.Input(shape=(ntimes_input, nlev, nfeat))

    # 时间分布卷积捕捉垂直信息
    x = layers.TimeDistributed(layers.Conv1D(64, 3, padding='same', activation='relu'))(inp)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)

    # reshape为GRU可用 shape: (samples, time, lev*feat)
    x = layers.Reshape((ntimes_input, nlev*nfeat))(x)

    # Attention层
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    # GRU捕捉时间特征
    x = layers.GRU(128, return_sequences=False)(x)

    # 输出
    outputs = layers.Dense(nlev*out_vars, activation='linear')(x)
    outputs = layers.Reshape((nlev, out_vars))(outputs)

    model = layers.Model(inputs=inp, outputs=outputs)
    return Model(inp, outputs, name="2d_conv")

def weighted_mse(y_true, y_pred):
    nlev = tf.shape(y_true)[1]
    weights = tf.linspace(2.0, 1.0, nlev)[:, tf.newaxis]  # shape: (lev,1)
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))
