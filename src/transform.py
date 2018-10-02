# coding: utf-8
import tensorflow as tf, pdb

WEIGHTS_INIT_STDEV = .1

def net(image):
    # 先执行3层卷积，每次卷积都会对结果归一化
    # 第一个参数是图像数据，第二是滤波器的数量，第三是滤波器尺寸，第四是滤波步长
    # 卷积方式全部采用的SAME，即在边界补0
    conv1 = _conv_layer(image, 32, 9, 1) # 输入尺寸 W×H，输出尺寸 W×H
    conv2 = _conv_layer(conv1, 64, 3, 2) # 输入尺寸 W×H，输出尺寸 （W/2）×（H/2）
    conv3 = _conv_layer(conv2, 128, 3, 2) # 输入尺寸（W/2）×（H/2），输出尺寸（W/4）×（H/4）

    # 执行5个残差卷积
    # 残差模块里使用的卷积方式SAME，stride为1,所以不改变特征图维度
    resid1 = _residual_block(conv3, 3)  # 输入尺寸（W/4）×（H/4），输出尺寸（W/4）×（H/4）
    resid2 = _residual_block(resid1, 3)
    resid3 = _residual_block(resid2, 3)
    resid4 = _residual_block(resid3, 3)
    resid5 = _residual_block(resid4, 3) # 输入尺寸（W/4）×（H/4），输出尺寸（W/4）×（H/4）

    # 执行2个反卷积层
    # 第一个参数输入img，第二滤波器数量，第三滤波器大小，第四滤波步长
    # 输出维度扩充stride=2倍
    conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2) # 输入尺寸（W/4）×（H/4）,输出尺寸（W×2/4）×（H×2/4）
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2)  # 输入尺寸（W×2/4）×（H×2/4）,输出尺寸（W×2×2/4）×（H×2×2/4）

    # 执行1个卷积
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False) # 输出尺寸W×H，输出尺寸W×H
    # 执行T激活函数，tensor的维度不变，结果有可能超[0，255]上下限，最后保存时候会做一个clip操作
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    return preds

'''
1. 产生网络随机数
2. 执行卷积
3. 卷积结果归一化
# 第一个参数是图像数据，第二是滤波器的数量，第三是滤波器尺寸，第四是滤波步长
'''
def _conv_layer(net, num_filters, filter_size, strides, relu=True):
    # 产生满足正太分布的随机初始参数
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    # 输入和第一层卷积层执行卷积，strdies=1
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    # 数据归一化
    net = _instance_norm(net)
    if relu:
        net = tf.nn.relu(net)
    return net

# 反卷积层，提升数据维度
# 第一个参数输入img，第二滤波器数量，第三滤波器大小，第四滤波步长
def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)
    # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    # tensor拼接
    tf_shape = tf.stack(new_shape)
    strides_shape = [1,strides,strides,1]

    #反卷积，提升维度为tf.shape
    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
    #数据归一化
    net = _instance_norm(net)
    return tf.nn.relu(net)

# 残差模块，当前层卷积，并且当前输入直接叠加到下一层
def _residual_block(net, filter_size=3):
    tmp = _conv_layer(net, 128, filter_size, 1)
    # tensor 相加，不改变维度
    return net + _conv_layer(tmp, 128, filter_size, 1, relu=False)

# 归一化，白化处理
def _instance_norm(net, train=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    # 计算统计矩，mu是均值，sigma_sq是方差
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

#根据输入的tensor维度产生截断的正太分布随机数
def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    return weights_init
