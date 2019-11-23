import model
import utils
import data_processing

def main():
    dataFilePath = ['data/ANightmareonElmStreet/data.npy', 'data/FinalFantasy/data.npy', 'data/WizardsAndWarriors/data.npy']
    labelFilePath = ['data/ANightmareonElmStreet/label.npy', 'data/FinalFantasy/label.npy', 'data/WizardsAndWarriors/label.npy']
    testset_ratio = 0.15
    validset_ratio = 0.02

    data_manager = data_processing.Preprocessing(dataFilePath, labelFilePath, testset_ratio, validset_ratio)

    X_train, Y_train = data_manager.get_train_data()
    X_val, Y_val = data_manager.get_val_data()
    X_test, Y_test = data_manager.get_test_data()

    # Example param for model
    model_params = ParamDict(
        model = 'default',               # use 'default' to init default model
        input_shape = (256, 256, 3),     # (length, width, channel)
        lr = 5e-4,
        step_rate = 500,
        decay_rate = 0.96,
        num_epoch = 5,
        batch_size = 64,
        log_step = 50,
    )

    tf.reset_default_graph()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        with tf.device('/cpu:0'):
            model = FeatureExtractionModel(model_params)
            model.train(sess, X_train, Y_train, X_val, Y_val)
            accuracy = model.evaluate(sess, X_test, Y_test)
            print('***** test accuracy: %.3f' % accuracy)
            saver = tf.train.Saver()
            model_path = saver.save(sess, "tf_models/feature_extraction_v1.ckpt")
            print("Model saved in %s" % model_path)


if __name__ == "__main__":
    main()
