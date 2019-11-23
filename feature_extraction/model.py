import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def conv2d(input, kernel_size, stride, num_filter):
    stride_shape = [1, stride, stride, 1]
    filter_shape = [kernel_size, kernel_size, input.get_shape()[3], num_filter]

    W = tf.compat.v1.get_variable('w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))
    b = tf.compat.v1.get_variable('b', [1, 1, 1, num_filter], initializer=tf.constant_initializer(0.0))
    return tf.nn.conv2d(input, W, stride_shape, padding='SAME') + b

def max_pool(input, kernel_size, stride):
    ksize = [1, kernel_size, kernel_size, 1]
    strides = [1, stride, stride, 1]
    return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding='SAME')

def flatten(input):
    return tf.layers.flatten(input)

def fc(input, num_output):
    num_input = input.shape[1]
    W = tf.compat.v1.get_variable('fc_w_%d' % num_output, [num_input, num_output], tf.float32, tf.random_normal_initializer(0.0, 0.02))
    b = tf.compat.v1.get_variable('fc_b_%d' % num_output, [num_output], initializer=tf.constant_initializer(0.0))
    return tf.nn.xw_plus_b(input, W, b, 'fc_%d' % num_output)


class FeatureExtractionModel(object):
    def __init__(self, param):
        """
        Example param:

            model: use 'default' to init default model
            input_shape: (length, width, channel)
            lr: 5e-4
            step_rate: 500
            decay_rate: 0.96
            num_epoch: 5
            batch_size: 64
            log_step: 50
        """
        self.param = param
        self._build_model()

    def _default_model(self):
        print('intput layer: ' + str(self.X.get_shape()))
        with tf.compat.v1.variable_scope('conv1'):
            self.conv1 = conv2d(self.X, 3, 1, 32)
            self.elu1 = tf.nn.elu(self.conv1)
            print('conv1 layer: ' + str(self.elu1.get_shape()))

        with tf.compat.v1.variable_scope('conv2'):
            self.conv2 = conv2d(self.elu1, 3, 1, 32)
            self.elu2 = tf.nn.elu(self.conv2)
            self.pool2 = max_pool(self.elu2, 2, 2)
            self.dropout2 = tf.layers.dropout(self.pool2, rate=0.4, training=self.training)
            print('conv2 layer: ' + str(self.dropout2.get_shape()))
        
        with tf.compat.v1.variable_scope('conv3'):
            self.conv3 = conv2d(self.dropout2, 3, 1, 64)
            self.elu3 = tf.nn.elu(self.conv3)
            print('conv3 layer: ' + str(self.elu3.get_shape()))
        
        with tf.compat.v1.variable_scope('conv4'):
            self.conv4 = conv2d(self.elu3, 3, 1, 64)
            self.elu4 = tf.nn.elu(self.conv4)
            self.pool4 = max_pool(self.elu4, 2, 2)
            self.dropout4 = tf.layers.dropout(self.pool4, rate=0.4, training=self.training)
            print('conv4 layer: ' + str(self.dropout4.get_shape()))
        
        with tf.compat.v1.variable_scope('conv5'):
            self.conv5 = conv2d(self.dropout4, 3, 1, 128)
            self.elu5 = tf.nn.elu(self.conv5)
            print('conv5 layer: ' + str(self.elu5.get_shape()))
        
        with tf.compat.v1.variable_scope('conv6'):
            self.conv6 = conv2d(self.elu5, 3, 1, 128)
            self.elu6 = tf.nn.elu(self.conv6)
            self.pool6 = max_pool(self.elu6, 2, 2)
            self.dropout6 = tf.layers.dropout(self.pool6, rate=0.4, training=self.training)
            print('conv6 layer: ' + str(self.dropout6.get_shape()))

        self.flat = flatten(self.dropout6)
        print('flat layer: ' + str(self.flat.get_shape()))
        
        with tf.compat.v1.variable_scope('fc7'):
            self.fc7 = fc(self.flat, 20)
            print('fc7 layer: ' + str(self.fc7.get_shape()))

        return self.fc7

    def _pretrain_model(self, name):
        # TODO add pretrain model such as AlexNet
        pass

    def _build_model(self):
        # Define input variables
        x_shape = [None]
        x_shape.extend(self.param.input_shape)
        self.X = tf.compat.v1.placeholder(tf.float32, x_shape)
        self.Y = tf.compat.v1.placeholder(tf.int64, [None])

        self.training = tf.compat.v1.placeholder(tf.bool)

        # Output is energy, which value will be [0, 19]
        labels = tf.one_hot(self.Y, 20)

        # Build a model and get logits
        if self.param.model is 'default':
            logits = self._default_model()
        else:
            logits = self._pretrain_model(self.param.model)

        # Compute loss
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels, logits))
        
        # Build optimizer
        lr = self.param.lr
        step_rate = self.param.step_rate
        decay_rate = self.param.decay_rate

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr, global_step, step_rate, decay_rate)

        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_op)

        # Compute accuracy
        predict = tf.argmax(logits, 1)
        correct = tf.equal(predict, self.Y)
        self.accuracy_op = tf.reduce_mean(tf.cast(correct, tf.float32))
        
    def train(self, sess, X_train, Y_train, X_val, Y_val):
        sess.run(tf.global_variables_initializer())

        step = 0
        losses = []
        accuracies = []
        print('-' * 5 + '  Start training  ' + '-' * 5)
        for epoch in range(self.param.num_epoch):
            print('train for epoch %d' % epoch)
            for i in range(self.param.num_training // self.param.batch_size):
                X_ = X_train[i * self.param.batch_size:(i + 1) * self.param.batch_size][:]
                Y_ = Y_train[i * self.param.batch_size:(i + 1) * self.param.batch_size]

                feed_dict = {self.X: X_, self.Y: Y_, self.training: True}

                fetches = [self.train_op, self.loss_op, self.accuracy_op]

                _, loss, accuracy = sess.run(fetches, feed_dict=feed_dict)
                losses.append(loss)
                accuracies.append(accuracy)

                if step % self.param.log_step == 0:
                    print('iteration (%d): loss = %.3f, accuracy = %.3f' %
                        (step, loss, accuracy))
                step += 1

            # Print validation results
            print('validation for epoch %d' % epoch)
            val_accuracy = self.evaluate(sess, X_val, Y_val)
            print('-  epoch %d: validation accuracy = %.3f' % (epoch, val_accuracy))
            
        # Graph 1. X: iteration (training step), Y: training loss
        plt.subplot(2, 1, 1)
        plt.title('Training loss')
        plt.plot(losses, '-o')
        plt.xlabel('IterationForLoss')
        # Graph 2. X: iteration (training step), Y: training accuracy
        plt.subplot(2, 1, 2)
        plt.title('Accuracy')
        plt.plot(accuracies, '-o')
        plt.xlabel('IterationForAccuracy')
        plt.gcf().set_size_inches(15, 12)
        plt.show()

    def evaluate(self, sess, X_eval, Y_eval):
        eval_accuracy = 0.0
        eval_iter = 0
        for i in range(X_eval.shape[0] // self.batch_size):
            X_ = X_eval[i * self.batch_size:(i + 1) * self.batch_size][:]
            Y_ = Y_eval[i * self.batch_size:(i + 1) * self.batch_size]

            feed_dict = {self.X: X_, self.Y: Y_, self.training: False}

            accuracy = sess.run(self.accuracy_op, feed_dict=feed_dict)
            eval_accuracy += accuracy
            eval_iter += 1
        return eval_accuracy / eval_iter
