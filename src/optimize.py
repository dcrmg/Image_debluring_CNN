# coding: utf-8
from __future__ import print_function
import functools
import vgg, pdb, time
import tensorflow as tf, numpy as np, os
import transform
from utils import get_img

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = ('relu3_2', 'relu4_2', 'relu4_2')
# CONTENT_LAYER = ['relu4_2']
DEVICES = 'CUDA_VISIBLE_DEVICES'

# np arr, np arr
def optimize(content_targets,clear_content_targets, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1000,
             batch_size=4, save_path='saver/fns.ckpt',learning_rate=1e-3, debug=False):
    mod = len(content_targets) % batch_size
    if mod > 0:
        print("Train set has been trimmed slightly..")
        content_targets = content_targets[:-mod]
        clear_content_targets = clear_content_targets[:-mod]

    batch_shape = (batch_size,256,256,3)


    with tf.Graph().as_default(), tf.Session() as sess:

        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
        clear_content = tf.placeholder(tf.float32, shape=batch_shape, name="clear_content")


        # transform.net定义了一个从内容图像生成合成图像的网络
        preds = transform.net(X_content/255.0)
        preds_pre = vgg.preprocess(preds)


        # 由模糊图像生成的清晰图像的VGG特征
        net = vgg.net(vgg_path, preds_pre)

        clear_pre = vgg.preprocess(clear_content)
        # 真实清晰图像的VGG特征
        clear_net = vgg.net(vgg_path,clear_pre)


        content_losses = []
        # clear img
        for content_layer in CONTENT_LAYER:

            content_features = clear_net[content_layer]
            content_size = _tensor_size(content_features)
            content_loss_ = (2 * tf.nn.l2_loss(net[content_layer] -
                                                    clear_net[content_layer]) / content_size)
            content_losses.append(content_loss_)

        # 计算内容图像和合成图像的l2损失
        content_loss = content_weight * functools.reduce(tf.add, content_losses) / batch_size

        style_losses = []

        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            clear_layer = clear_net[style_layer]

            bs, height, width, filters = map(lambda i:i.value,layer.get_shape())
            clear_bs, clear_height, clear_width, clear_filters = map(lambda i:i.value,clear_layer.get_shape())
            size = height * width * filters
            clear_size = clear_height * clear_width * clear_filters

            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0,2,1])

            clear_feats = tf.reshape(clear_layer, (clear_bs, clear_height * clear_width, clear_filters))
            clear_feats_T = tf.transpose(clear_feats, perm=[0,2,1])

            grams = tf.matmul(feats_T, feats) / size
            clear_grams = tf.matmul(clear_feats_T, clear_feats) / clear_size
            # style_gram = style_features[style_layer]
            gram_size = grams.get_shape()[1]*grams.get_shape()[-1]
            style_losses.append(2 * tf.nn.l2_loss(grams - clear_grams)/int(gram_size))

        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size

        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])

        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])        
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size


        loss = content_loss + style_loss + tv_loss


        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())

        # writer = tf.summary.FileWriter('./log/', sess.graph)

        print("Training begins!")

        for epoch in range(epochs):

            save_flag = True
            num_examples = len(content_targets)
            iterations = 0
            while iterations * batch_size < num_examples:
                start_time = time.time()
                curr = iterations * batch_size
                step = curr + batch_size
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                clear_X_batch = np.zeros(batch_shape, dtype=np.float32)

                for j, img_p in enumerate(content_targets[curr:step]):
                   X_batch[j] = get_img(img_p, (256,256,3)).astype(np.float32)

                for j, img_p in enumerate(clear_content_targets[curr:step]):
                    clear_X_batch[j] = get_img(img_p, (256, 256, 3)).astype(np.float32)

                iterations += 1
                assert X_batch.shape[0] == batch_size

                feed_dict = {
                   X_content:X_batch,clear_content:clear_X_batch
                }

                train_step.run(feed_dict=feed_dict)
                end_time = time.time()
                delta_time = end_time - start_time

                if debug:
                    loss_ = loss
                    test_feed_dict = {
                        X_content: X_batch, clear_content: clear_X_batch
                    }

                    tup = sess.run(loss_, feed_dict=test_feed_dict)
                    print("epoch: %s, loss: %s, batch time: %s" % (epoch, tup, delta_time))
                is_print_iter = int(iterations) % print_iterations == 0
                is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples
                should_print = is_print_iter or is_last or (epoch+1)%1==0
                if should_print and save_flag:
                    save_flag =False
                    to_get = [style_loss, content_loss, tv_loss, loss, preds]
                    test_feed_dict = {
                        X_content: X_batch, clear_content: clear_X_batch
                    }

                    tup = sess.run(to_get, feed_dict = test_feed_dict)
                    _style_loss,_content_loss,_tv_loss,_loss,_preds = tup
                    losses = (_style_loss, _content_loss, _tv_loss, _loss)

                    saver = tf.train.Saver()
                    res = saver.save(sess, save_path+'{}_{}.ckpt'.format(epoch,int(_content_loss)), write_meta_graph=False)
                    yield(_preds, losses, iterations, epoch)

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)
