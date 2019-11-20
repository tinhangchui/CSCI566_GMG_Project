def main:
    num_dev_data = 50
    num_training = 49
    num_validation = num_dev_data - num_training
    num_test = 10

    # TODO: data processing
    data = prepocessing()

    X_train, Y_train = get_train_data(data)
    X_val, Y_val = get_val_data(data)
    X_test, Y_test = get_test_data(data)

    tf.reset_default_graph()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        with tf.device('/cpu:0'):
            model = FeatureExtractionModel()
            model.train(sess, X_train, Y_train, X_val, Y_val)
            accuracy = model.evaluate(sess, X_test, Y_test)
            print('***** test accuracy: %.3f' % accuracy)
            saver = tf.train.Saver()
            model_path = saver.save(sess, "tf_models/feature_extraction_v1.ckpt")
            print("Model saved in %s" % model_path)


if __name__ == "__main__":
    main()