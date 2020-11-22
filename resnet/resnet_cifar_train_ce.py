#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   resnet_cifar_train_ce.py
@Contact :   liangyacongyi@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2019/3/15 11:28 AM   liangcong      1.0   Train ResNet model with CIFAR-10/100

run this file with
"python resnet_cifar_train_ce.py --data_type=cifar_10 --num_blocks=3 --weight_decay=0.0001"
"""
import numpy as np
import tensorflow as tf
import utils as utl
import datetime
import os

from sklearn.metrics import confusion_matrix
from data import get_data_set
from data_aug import torch_input
from resnet_cifar_model import model
tf.set_random_seed(1234)
# ------------------------------command line argument---------------------------------
tf.app.flags.DEFINE_float("weight_decay", 0.0001, "weight decay, "
                                                  "(ResNet-20: 0.0001, ResNet-32: 0.0002 ,ResNet-44: 0.0003"
                                                  "ResNet-56: 0.0004, ResNet-110: 0.0005)")
tf.app.flags.DEFINE_string("data_type", 'cifar_10', "cifar_10 or cifar_100, default:cifar_10")
tf.app.flags.DEFINE_integer("num_blocks", 3, "layers number: 6*num_blocks+2, selected from{3,5,7,9,18}")
FLAGS = tf.app.flags.FLAGS
_DATA_TYPE = FLAGS.data_type
_WEIGHT_DECAY = FLAGS.weight_decay
_NUM_BLOCKS = FLAGS.num_blocks
# ------------------------------------------------------------------------------------

# ------------------------------command line argument judge---------------------------
if not isinstance(_DATA_TYPE, str):
    print("data_type must be a string")
    os._exit(0)
if not isinstance(_NUM_BLOCKS, int):
    print("num_blocks must be a int number")
    os._exit(0)
    
if _DATA_TYPE not in ('cifar_10', 'cifar_100'):
    print("_DATA_TYPE must be cifar_10 or cifar_100")
    os._exit(0)
if _NUM_BLOCKS not in (3, 5, 7, 9, 18):
    print("_DATA_TYPE must be selected from {3,5,7,9,18}")
    os._exit(0)
# ------------------------------------------------------------------------------------

# ------------------------------fixed parameters--------------------------------------
_ITERATION = 64000
_BATCH_SIZE = 128
_CLASS_SIZE = int(_DATA_TYPE.split('_')[1])
_vec = np.ones(_CLASS_SIZE, 'float32')
# ------------------------------------------------------------------------------------

print("num_layers: %d, data_type: %s" % (6*FLAGS.num_blocks+2, _DATA_TYPE))

# ------------------------------create relative folders-------------------------------
PAR_PATH = "./" + _DATA_TYPE + "/" + str(6*_NUM_BLOCKS+2) + "/ce"
_MODEL_SAVE_PATH = os.path.join(PAR_PATH, "model/")
if not os.path.exists(PAR_PATH):
    os.makedirs(PAR_PATH)
if not os.path.exists(_MODEL_SAVE_PATH):
    os.makedirs(_MODEL_SAVE_PATH)
_TENSORBOARD_SAVE_PATH = os.path.join(PAR_PATH, "tensorboard")
# ------------------------------------------------------------------------------------

x, y, output, global_step, y_pred_cls, c, phase_train = model(_CLASS_SIZE, _NUM_BLOCKS)

# ------------------------------data pre-process-------------------------------------
train_x, train_y, train_l = get_data_set(name="train", cifar=_CLASS_SIZE)
test_x, test_y, test_l = get_data_set(name="test", cifar=_CLASS_SIZE)

print("mean subtracted")
mean = np.mean(train_x, axis=1)
train_x = train_x - mean[:, np.newaxis]
test_mean = np.mean(test_x, axis=1)
test_x = test_x - test_mean[:, np.newaxis]
print("mean subtracted end")
# ------------------------------------------------------------------------------------

# ------------------------------L_{SE} loss-------------------------------------------
l_se = utl.secondary_optimal_feature_plane_loss(output, y, _vec)
# ------------------------------------------------------------------------------------

# ------------------------------final loss--------------------------------------------
ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
l_pe = ce_loss
loss = l_pe
# ------------------------------------------------------------------------------------

# ------------------------------weight regularization---------------------------------
weight_decay = _WEIGHT_DECAY
t_v = tf.losses.get_regularization_losses()
w12 = tf.add_n([t_v[i] for i in range(len(t_v))])
wl2_loss = w12 * weight_decay
# ------------------------------------------------------------------------------------

# ------------------------------optimal algorithm setting-----------------------------
if _NUM_BLOCKS == 18:
    boundaries = [400, 32000, 48000]  # for layer-110
    values = [0.01, 0.1, 0.01, 0.001]
else:
    boundaries = [32000, 48000]
    values = [0.1, 0.01, 0.001]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
train_op = tf.train.MomentumOptimizer(learning_rate, 0.9, name='Momentum1', use_nesterov=True)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = train_op.minimize(loss + wl2_loss, global_step=global_step)
# ------------------------------------------------------------------------------------

correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, dimension=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("Accuracy/train", accuracy)
tf.summary.scalar("Loss", loss)

merged = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=1)
sess = tf.Session()
train_writer = tf.summary.FileWriter(_TENSORBOARD_SAVE_PATH, sess.graph)
sess.run(tf.global_variables_initializer())


def train(itera, f):
    """
    Train CNN
    :param itera: numbers of iteration
    :return:
    """
    global train_x
    global train_y

    _pe_loss = []
    _se_loss = []
    _test_acc = []
    _cm = []
    _feature = []

    print('start training')
    print >> f, 'start training'

    print("parameter settings, num_layers: %d, data_type: %s" % (6*FLAGS.num_blocks+2, _DATA_TYPE))
    print >> f, "parameter settings, num_layers: %d, data_type: %s" % (6*FLAGS.num_blocks+2, _DATA_TYPE)

    for i in range(itera):
        randidx = np.random.choice(len(train_x), size=_BATCH_SIZE, replace=False)
        batch_xs = torch_input(train_x[randidx])
        batch_ys = train_y[randidx]

        i_global, _, l_loss, l_pe_loss, l_se_loss, l_acc = \
            sess.run([global_step, optimizer, loss, l_pe, l_se, accuracy],
                     feed_dict={x: batch_xs, y: batch_ys, phase_train: True})

        if i_global % 10 == 0:
            print("Global step: %d, batch accuracy: %.4f, loss: %.4f, pe loss: %.4f, se loss: %.4f"
                  % (i_global, l_acc, l_loss, l_pe_loss, l_se_loss))
            print >> f, "Global step: %d, batch accuracy: %.4f, loss: %.4f, pe loss: %.4f, se loss: %.4f" \
                        % (i_global, l_acc, l_loss, l_pe_loss, l_se_loss)

        if i_global % 1000 == 0 or i_global == itera:
            data_merged = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, phase_train: True})
            print("Global step: %d, batch accuracy: %.4f, loss: %.4f, pe loss: %.4f, se loss: %.4f"
                  % (i_global, l_acc, l_loss, l_pe_loss, l_se_loss))
            print >> f, "Global step: %d, batch accuracy: %.4f, loss: %.4f, pe loss: %.4f, se loss: %.4f" \
                        % (i_global, l_acc, l_loss, l_pe_loss, l_se_loss)

            acc, cm, feature = predict_test(f)

            _test_acc.append(acc)
            _pe_loss.append(l_pe_loss)
            _se_loss.append(l_se_loss)

            summary = tf.Summary(value=[
                tf.Summary.Value(tag="Accuracy/test", simple_value=acc)])
            train_writer.add_summary(data_merged, i_global)
            train_writer.add_summary(summary, i_global)

            if i_global == 1000:
                saver.save(sess, save_path=_MODEL_SAVE_PATH, global_step=global_step)
                print("Saved checkpoint.")
                print >> f, "Saved checkpoint."
                _temp_acc = acc
                _temp_cm = cm
                _temp_feature = feature

            if acc > _temp_acc:
                saver.save(sess, save_path=_MODEL_SAVE_PATH, global_step=global_step)
                print("Saved checkpoint.")
                print >> f, "Saved checkpoint."
                _temp_acc = acc
                _temp_cm = cm
                _temp_feature = feature

    _feature.append(_temp_feature)
    _cm.append(cm)

    return _pe_loss, _se_loss, _test_acc, _cm, _feature


def predict_test(f, show_confusion_matrix=False):
    """
    Make prediction for all images in test_x
    :param show_confusion_matrix: default false
    :return: accuracy
    """
    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    feature = np.zeros(shape=(len(test_x), _CLASS_SIZE), dtype=np.float)
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j], feature[i:j, :] = sess.run([y_pred_cls, output],
                                                         feed_dict={x: batch_xs, y: batch_ys, phase_train: False})
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean() * 100
    correct_numbers = correct.sum()
    print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))
    print >> f, ("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))

    cm = confusion_matrix(y_true=np.argmax(test_y, axis=1), y_pred=predicted_class)
    if show_confusion_matrix is True:
        cm = confusion_matrix(y_true=np.argmax(test_y, axis=1), y_pred=predicted_class)
        for i in range(_CLASS_SIZE):
            class_name = "({}) {}".format(i, test_l[i])
            print(cm[i, :], class_name)
        class_numbers = [" ({0})".format(i) for i in range(_CLASS_SIZE)]
        print("".join(class_numbers))

    return acc, cm, feature


def main(_):
    f = open(os.path.join(PAR_PATH, "train_info_.txt"), "w")
    time1 = datetime.datetime.now()
    pe, se, test_acc, cm, feature = train(_ITERATION, f)
    time2 = datetime.datetime.now()
    duration = time2 - time1
    print "duration:", duration
    print("the best test_acc: %.3f, in epoch %d" % (np.max(test_acc), np.argmax(test_acc)))
    print >> f, "duration:", duration
    print >> f, "the best test_acc: %.3f, in epoch %d" % (np.max(test_acc), np.argmax(test_acc))
    f.close()

    np.savetxt(os.path.join(PAR_PATH, "pe_loss.txt"), pe)
    np.savetxt(os.path.join(PAR_PATH, "se_loss.txt"), se)
    np.savetxt(os.path.join(PAR_PATH, "test_acc.txt"), test_acc)
    np.save(os.path.join(PAR_PATH, "cm.npy"), cm)
    np.save(os.path.join(PAR_PATH, "feature.npy"), feature)


if __name__ == "__main__":
    tf.app.run()

sess.close()


