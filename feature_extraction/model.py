import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


def mkdir(folder_name, loss_name):
    prefix = os.path.dirname(os.path.abspath("__file__"))
    path = prefix + "\\tf_models\\" + folder_name
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print ("--->  build new folder: " + path + " <---")
    return path

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

        with tf.compat.v1.variable_scope('fc_tse1'):
            self.fc_tse1 = fc(self.flat, 256)
            print('fc_tse1 layer: ' + str(self.fc_tse1.get_shape()))

        with tf.compat.v1.variable_scope('fc_tse2'):
            self.fc_tse2 = fc(self.fc_tse1, 256)
            print('fc_tse2 layer: ' + str(self.fc_tse2.get_shape()))

        with tf.compat.v1.variable_scope('fc_tse3'):
            self.fc_tse3 = fc(self.fc_tse2, 256)
            print('fc_tse3 layer: ' + str(self.fc_tse3.get_shape()))

        with tf.compat.v1.variable_scope('fc_tse'):
            self.fc_tse = fc(self.fc_tse3, 3)
            print('fc_tse layer: ' + str(self.fc_tse.get_shape()))

        with tf.compat.v1.variable_scope('fc_bpm'):
            self.fc_bpm = fc(self.flat, 1)
            print('fc_bpm layer: ' + str(self.fc_bpm.get_shape()))

        with tf.compat.v1.variable_scope('fc_energy1'):
            self.fc_energy1 = fc(self.flat, 256)
            print('fc_energy1 layer: ' + str(self.fc_energy1.get_shape()))

        with tf.compat.v1.variable_scope('fc_energy2'):
            self.fc_energy2 = fc(self.fc_energy1, 256)
            print('fc_energy2 layer: ' + str(self.fc_energy2.get_shape()))

        with tf.compat.v1.variable_scope('fc_energy3'):
            self.fc_energy3 = fc(self.fc_energy2, 256)
            print('fc_energy3 layer: ' + str(self.fc_energy3.get_shape()))

        with tf.compat.v1.variable_scope('fc_energy'):
            self.fc_energy = fc(self.fc_energy3, 20)
            print('fc_energy layer: ' + str(self.fc_energy.get_shape()))

        return [self.fc_tse, self.fc_bpm, self.fc_energy]

    def _pretrain_model(self, name):
        # TODO add pretrain model such as AlexNet
        pass

    def _build_model(self):
        # Define input variables
        x_shape = [None]
        y_shape = [None]
        x_shape.extend(self.param.input_shape)
        y_shape.append(self.param.output_dimension)
        self.X = tf.compat.v1.placeholder(tf.float32, x_shape)
        if self.param.loss_name is "bpm_loss":
            self.Y = tf.compat.v1.placeholder(tf.float32, y_shape)
        else:
            self.Y = tf.compat.v1.placeholder(tf.int64, y_shape)

        self.training = tf.compat.v1.placeholder(tf.bool)

        if self.param.loss_name is "bpm_loss":
            # Output is bpm, value in [0, 600?]
            bpm_lables = self.Y[:, 1]
        else:
            # Output is tse, which value will be [0, 2]
            tse_labels = tf.one_hot(self.Y[:,0], 3)
            # Output is energy, which value will be [0, 19]
            energy_labels = tf.one_hot(self.Y[:,2], 20)

        # Build a model and get logits
        if self.param.model is 'default':
            logits_tse, regression_bpm, logits_energy = self._default_model()
        else:
            logits = self._pretrain_model(self.param.model)

        # Compute loss
        if self.param.loss_name is "bpm_loss":
            bpm_loss = tf.reduce_mean(tf.nn.l2_loss(bpm_lables - regression_bpm))
        else:
            tse_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(tse_labels, logits_tse))
            energy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(energy_labels, logits_energy))

        if self.param.loss_name == "energy_loss":
            print('Energy loss')
            self.loss_op = energy_loss
        elif self.param.loss_name == "tse_loss":
            print('Tse loss')
            self.loss_op = tse_loss
        else:
            self.loss_op = bpm_loss


        # Build optimizer
        lr = self.param.lr
        step_rate = self.param.step_rate
        decay_rate = self.param.decay_rate

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr, global_step, step_rate, decay_rate)

        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_op)

        if self.param.loss_name is not "bpm_loss":
            # Compute accuracy
            # tse
            self.tse_predict_op = tf.argmax(logits_tse, 1)
            tse_correct = tf.equal(self.tse_predict_op, self.Y[:,0])
            self.tse_accuracy_op = tf.reduce_mean(tf.cast(tse_correct, tf.float32))
            # energy
            self.energy_predict_op = tf.argmax(logits_energy, 1)
            energy_correct = tf.equal(self.energy_predict_op, self.Y[:,2])
            self.energy_accuracy_op = tf.reduce_mean(tf.cast(energy_correct, tf.float32))
        
    def train(self, sess, X_train, Y_train, X_val, Y_val):
        sess.run(tf.global_variables_initializer())

        step = 0
        losses = []
        tse_accuracies = []
        energy_accuracies = []
        print('-' * 5 + '  Start training  ' + '-' * 5)
        for epoch in range(self.param.num_epoch):
            print('train for epoch %d' % epoch)
            for i in range(self.param.num_training // self.param.batch_size):
                X_ = X_train[i * self.param.batch_size:(i + 1) * self.param.batch_size][:]
                Y_ = Y_train[i * self.param.batch_size:(i + 1) * self.param.batch_size]

                feed_dict = {self.X: X_, self.Y: Y_, self.training: True}

                if self.param.loss_name is not "bpm_loss":
                    fetches = [self.train_op, self.loss_op, self.tse_accuracy_op, self.energy_accuracy_op]

                    _, loss, tse_accuracy, energy_accuracy = sess.run(fetches, feed_dict=feed_dict)
                    tse_accuracies.append(tse_accuracy)
                    energy_accuracies.append(energy_accuracy)
                else:
                    fetches = [self.train_op, self.loss_op]
                    _, loss = sess.run(fetches, feed_dict=feed_dict)

                losses.append(loss)

                if step % self.param.log_step == 0:
                    if self.param.loss_name is not "bpm_loss":
                        print('iteration (%d): loss = %.3f, tse_accuracy = %.3f, energy_accuracy = %.3f' %
                            (step, loss, tse_accuracy, energy_accuracy))
                    else:
                        print('iteration (%d): loss = %.3f' %
                              (step, loss))
                step += 1

            if self.param.loss_name is not "bpm_loss":
                # Print validation results
                print('validation for epoch %d' % epoch)
                val_tse_accuracy, val_energy_accuracy = self.evaluate(sess, X_val, Y_val)
                print('-  epoch %d: validation tse_accuracy = %.3f, energy_accuracy = %.3f' % (epoch, val_tse_accuracy, val_energy_accuracy))
            
        # Graph 1. X: iteration (training step), Y: training loss
        plt.subplot(2, 1, 1)
        plt.title('Training loss')
        plt.plot(losses, '-o')
        plt.xlabel('IterationForLoss')
        # Graph 2. X: iteration (training step), Y: training accuracy
        plt.subplot(2, 1, 2)
        plt.title('Accuracy')
        # TODO: revise plot accuracies
        if self.param.loss_name is "tse_loss":
            plt.plot(tse_accuracies, '-o')
        if self.param.loss_name is "energy_loss":
            plt.plot(energy_accuracies, '-o')
        plt.xlabel('IterationForAccuracy')
        plt.gcf().set_size_inches(15, 12)
        # save training process image
        prefix = mkdir(self.param.model_name, "whatever")
        plt.savefig(
            prefix + "//" + self.param.model_name + "_" + self.param.loss_name + ".png")

    def evaluate(self, sess, X_eval, Y_eval):
        eval_tse_accuracy = 0.0
        eval_energy_accuracy = 0.0
        eval_iter = 0
        for i in range(X_eval.shape[0] // self.param.batch_size):
            X_ = X_eval[i * self.param.batch_size:(i + 1) * self.param.batch_size][:]
            Y_ = Y_eval[i * self.param.batch_size:(i + 1) * self.param.batch_size]

            feed_dict = {self.X: X_, self.Y: Y_, self.training: False}

            tse_accuracy, energy_accuracy = sess.run([self.tse_accuracy_op, self.energy_accuracy_op], feed_dict=feed_dict)
            eval_tse_accuracy += tse_accuracy
            eval_energy_accuracy += energy_accuracy
            eval_iter += 1
        return eval_tse_accuracy / eval_iter, eval_energy_accuracy / eval_iter

    def predict(self, sess, X_predict):
        feed_dict = {self.X: X_predict, self.training: False}
        prediction = sess.run(self.predict_op, feed_dict=feed_dict)
        np.savetxt('predict.out', prediction)
        np.savetxt('predict_round.out', np.rint(prediction), fmt='%d')


class TseFeatureExtractionModel(object):
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

        with tf.compat.v1.variable_scope('fc_tse1'):
            self.fc_tse1 = fc(self.flat, 256)
            print('fc_tse1 layer: ' + str(self.fc_tse1.get_shape()))

        with tf.compat.v1.variable_scope('fc_tse2'):
            self.fc_tse2 = fc(self.fc_tse1, 256)
            print('fc_tse2 layer: ' + str(self.fc_tse2.get_shape()))

        with tf.compat.v1.variable_scope('fc_tse3'):
            self.fc_tse3 = fc(self.fc_tse2, 256)
            print('fc_tse3 layer: ' + str(self.fc_tse3.get_shape()))

        with tf.compat.v1.variable_scope('fc_tse'):
            self.fc_tse = fc(self.fc_tse3, 3)
            print('fc_tse layer: ' + str(self.fc_tse.get_shape()))

        return self.fc_tse

    def _pretrain_model(self, name):
        # TODO add pretrain model such as AlexNet
        pass

    def _build_model(self):
        # Define input variables
        x_shape = [None]
        y_shape = [None]
        x_shape.extend(self.param.input_shape)
        y_shape.append(self.param.output_dimension)
        self.X = tf.compat.v1.placeholder(tf.float32, x_shape)
        self.Y = tf.compat.v1.placeholder(tf.int64, y_shape)

        self.training = tf.compat.v1.placeholder(tf.bool)

        # Output is tse, which value will be [0, 2]
        tse_labels = tf.one_hot(self.Y[:, 0], 3)

        # Build a model and get logits
        logits_tse = self._default_model()

        # Compute loss
        tse_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(tse_labels, logits_tse))

        self.loss_op = tse_loss

        # Build optimizer
        lr = self.param.lr
        step_rate = self.param.step_rate
        decay_rate = self.param.decay_rate

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr, global_step, step_rate, decay_rate)

        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_op)

        # tse
        self.tse_predict_op = tf.argmax(logits_tse, 1)
        tse_correct = tf.equal(self.tse_predict_op, self.Y[:, 0])
        self.tse_accuracy_op = tf.reduce_mean(tf.cast(tse_correct, tf.float32))

    def train(self, sess, X_train, Y_train, X_val, Y_val):
        sess.run(tf.global_variables_initializer())

        step = 0
        losses = []
        tse_accuracies = []
        print('-' * 5 + '  Start training  ' + '-' * 5)
        for epoch in range(self.param.num_epoch):
            print('train for epoch %d' % epoch)
            for i in range(self.param.num_training // self.param.batch_size):
                X_ = X_train[i * self.param.batch_size:(i + 1) * self.param.batch_size][:]
                Y_ = Y_train[i * self.param.batch_size:(i + 1) * self.param.batch_size]

                feed_dict = {self.X: X_, self.Y: Y_, self.training: True}

                fetches = [self.train_op, self.loss_op, self.tse_accuracy_op]

                _, loss, tse_accuracy = sess.run(fetches, feed_dict=feed_dict)
                tse_accuracies.append(tse_accuracy)

                losses.append(loss)

                if step % self.param.log_step == 0:
                    print('iteration (%d): loss = %.3f, tse_accuracy = %.3f' % (step, loss, tse_accuracy))
                step += 1

            # Print validation results
            print('validation for epoch %d' % epoch)
            val_tse_accuracy = self.evaluate(sess, X_val, Y_val)
            print('-  epoch %d: validation tse_accuracy = %.3f' % (epoch, val_tse_accuracy))

        plt.figure(1)
        # Graph 1. X: iteration (training step), Y: training loss
        plt.subplot(2, 1, 1)
        plt.title('Training loss')
        plt.plot(losses, '-o')
        plt.xlabel('IterationForLoss')
        # Graph 2. X: iteration (training step), Y: training accuracy
        plt.subplot(2, 1, 2)
        plt.title('Accuracy')
        # TODO: revise plot accuracies
        plt.plot(tse_accuracies, '-o')
        plt.xlabel('IterationForAccuracy')
        plt.gcf().set_size_inches(15, 12)
        # plt.show()
        # save training process image
        prefix = mkdir(self.param.model_name, "tse_loss")
        plt.savefig(
            prefix + "//" + self.param.model_name + "_" + self.param.loss_name + ".png")

    def evaluate(self, sess, X_eval, Y_eval):
        eval_tse_accuracy = 0.0
        eval_iter = 0
        for i in range(X_eval.shape[0] // self.param.batch_size):
            X_ = X_eval[i * self.param.batch_size:(i + 1) * self.param.batch_size][:]
            Y_ = Y_eval[i * self.param.batch_size:(i + 1) * self.param.batch_size]

            feed_dict = {self.X: X_, self.Y: Y_, self.training: False}

            tse_accuracy = sess.run(self.tse_accuracy_op, feed_dict=feed_dict)
            eval_tse_accuracy += tse_accuracy
            eval_iter += 1
        return eval_tse_accuracy / eval_iter

    def predict(self, sess, X_predict):
        feed_dict = {self.X: X_predict, self.training: False}
        prediction = sess.run(self.tse_predict_op, feed_dict=feed_dict)
        return prediction


class BpmFeatureExtractionModel(object):
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

        with tf.compat.v1.variable_scope('fc_bpm'):
            self.fc_bpm = fc(self.flat, 1)
            print('fc_bpm layer: ' + str(self.fc_bpm.get_shape()))

        return self.fc_bpm

    def _pretrain_model(self, name):
        # TODO add pretrain model such as AlexNet
        pass

    def _build_model(self):
        # Define input variables
        x_shape = [None]
        y_shape = [None]
        x_shape.extend(self.param.input_shape)
        y_shape.append(self.param.output_dimension)
        self.X = tf.compat.v1.placeholder(tf.float32, x_shape)
        self.Y = tf.compat.v1.placeholder(tf.float32, y_shape)

        self.training = tf.compat.v1.placeholder(tf.bool)

        bpm_lables = self.Y[:, 1]

        # Build a model and get logits
        regression_bpm = self._default_model()

        # Compute loss
        bpm_loss = tf.reduce_mean(tf.nn.l2_loss(bpm_lables - regression_bpm))

        self.loss_op = bpm_loss

        # Build optimizer
        lr = self.param.lr
        step_rate = self.param.step_rate
        decay_rate = self.param.decay_rate

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr, global_step, step_rate, decay_rate)

        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_op)

        self.bpm_predict_op = regression_bpm

    def train(self, sess, X_train, Y_train, X_val, Y_val):
        sess.run(tf.global_variables_initializer())

        step = 0
        losses = []
        print('-' * 5 + '  Start training  ' + '-' * 5)
        for epoch in range(self.param.num_epoch):
            print('train for epoch %d' % epoch)
            for i in range(self.param.num_training // self.param.batch_size):
                X_ = X_train[i * self.param.batch_size:(i + 1) * self.param.batch_size][:]
                Y_ = Y_train[i * self.param.batch_size:(i + 1) * self.param.batch_size]

                feed_dict = {self.X: X_, self.Y: Y_, self.training: True}

                fetches = [self.train_op, self.loss_op]
                _, loss = sess.run(fetches, feed_dict=feed_dict)

                losses.append(loss)

                if step % self.param.log_step == 0:
                    print('iteration (%d): loss = %.3f' %
                          (step, loss))
                step += 1

        plt.figure(2)
        # Graph 1. X: iteration (training step), Y: training loss
        plt.subplot(2, 1, 1)
        plt.title('Training loss')
        plt.plot(losses, '-o')
        plt.xlabel('IterationForLoss')
        # Graph 2. X: iteration (training step), Y: training accuracy
        plt.subplot(2, 1, 2)
        plt.title('Accuracy')
        plt.gcf().set_size_inches(15, 12)
        # save training process image
        prefix = mkdir(self.param.model_name, "bpm_loss")
        plt.savefig(
            prefix + "//" + self.param.model_name + "_" + self.param.loss_name + ".png")

    def predict(self, sess, X_predict):
        feed_dict = {self.X: X_predict, self.training: False}
        prediction = sess.run(self.bpm_predict_op, feed_dict=feed_dict)
        return prediction


class EnergyFeatureExtractionModel(object):
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

        with tf.compat.v1.variable_scope('fc_energy1'):
            self.fc_energy1 = fc(self.flat, 256)
            print('fc_energy1 layer: ' + str(self.fc_energy1.get_shape()))

        with tf.compat.v1.variable_scope('fc_energy2'):
            self.fc_energy2 = fc(self.fc_energy1, 256)
            print('fc_energy2 layer: ' + str(self.fc_energy2.get_shape()))

        with tf.compat.v1.variable_scope('fc_energy3'):
            self.fc_energy3 = fc(self.fc_energy2, 256)
            print('fc_energy3 layer: ' + str(self.fc_energy3.get_shape()))

        with tf.compat.v1.variable_scope('fc_energy'):
            self.fc_energy = fc(self.fc_energy3, 20)
            print('fc_energy layer: ' + str(self.fc_energy.get_shape()))

        return self.fc_energy

    def _pretrain_model(self, name):
        # TODO add pretrain model such as AlexNet
        pass

    def _build_model(self):
        # Define input variables
        x_shape = [None]
        y_shape = [None]
        x_shape.extend(self.param.input_shape)
        y_shape.append(self.param.output_dimension)
        self.X = tf.compat.v1.placeholder(tf.float32, x_shape)
        self.Y = tf.compat.v1.placeholder(tf.int64, y_shape)

        self.training = tf.compat.v1.placeholder(tf.bool)

        # Output is energy, which value will be [0, 19]
        energy_labels = tf.one_hot(self.Y[:, 2], 20)

        # Build a model and get logits
        logits_energy = self._default_model()

        # Compute loss
        energy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(energy_labels, logits_energy))

        self.loss_op = energy_loss

        # Build optimizer
        lr = self.param.lr
        step_rate = self.param.step_rate
        decay_rate = self.param.decay_rate

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr, global_step, step_rate, decay_rate)

        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_op)

        # Compute accuracy
        # energy
        self.energy_predict_op = tf.argmax(logits_energy, 1)
        energy_correct = tf.equal(self.energy_predict_op, self.Y[:, 2])
        self.energy_accuracy_op = tf.reduce_mean(tf.cast(energy_correct, tf.float32))

    def train(self, sess, X_train, Y_train, X_val, Y_val):
        sess.run(tf.global_variables_initializer())

        step = 0
        losses = []
        energy_accuracies = []
        print('-' * 5 + '  Start training  ' + '-' * 5)
        for epoch in range(self.param.num_epoch):
            print('train for epoch %d' % epoch)
            for i in range(self.param.num_training // self.param.batch_size):
                X_ = X_train[i * self.param.batch_size:(i + 1) * self.param.batch_size][:]
                Y_ = Y_train[i * self.param.batch_size:(i + 1) * self.param.batch_size]

                feed_dict = {self.X: X_, self.Y: Y_, self.training: True}

                fetches = [self.train_op, self.loss_op, self.energy_accuracy_op]

                _, loss, energy_accuracy = sess.run(fetches, feed_dict=feed_dict)
                energy_accuracies.append(energy_accuracy)

                losses.append(loss)

                if step % self.param.log_step == 0:
                    print('iteration (%d): loss = %.3f, energy_accuracy = %.3f' %
                          (step, loss, energy_accuracy))
                step += 1

            # Print validation results
            print('validation for epoch %d' % epoch)
            val_energy_accuracy = self.evaluate(sess, X_val, Y_val)
            print('-  epoch %d: validation energy_accuracy = %.3f' % (epoch, val_energy_accuracy))

        plt.figure(3)
        # Graph 1. X: iteration (training step), Y: training loss
        plt.subplot(2, 1, 1)
        plt.title('Training loss')
        plt.plot(losses, '-o')
        plt.xlabel('IterationForLoss')
        # Graph 2. X: iteration (training step), Y: training accuracy
        plt.subplot(2, 1, 2)
        plt.title('Accuracy')
        plt.plot(energy_accuracies, '-o')
        plt.xlabel('IterationForAccuracy')
        plt.gcf().set_size_inches(15, 12)
        # save training process image
        prefix = mkdir(self.param.model_name, "energy_loss")
        plt.savefig(
            prefix + "//" + self.param.model_name + "_" + self.param.loss_name + ".png")

    def evaluate(self, sess, X_eval, Y_eval):
        eval_energy_accuracy = 0.0
        eval_iter = 0
        for i in range(X_eval.shape[0] // self.param.batch_size):
            X_ = X_eval[i * self.param.batch_size:(i + 1) * self.param.batch_size][:]
            Y_ = Y_eval[i * self.param.batch_size:(i + 1) * self.param.batch_size]

            feed_dict = {self.X: X_, self.Y: Y_, self.training: False}

            energy_accuracy = sess.run(self.energy_accuracy_op, feed_dict=feed_dict)
            eval_energy_accuracy += energy_accuracy
            eval_iter += 1
        return eval_energy_accuracy / eval_iter

    def predict(self, sess, X_predict):
        feed_dict = {self.X: X_predict, self.training: False}
        prediction = sess.run(self.energy_predict_op, feed_dict=feed_dict)
        return prediction
