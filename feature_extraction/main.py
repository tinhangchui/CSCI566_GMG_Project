import model
import utils
import data_processing
import tensorflow as tf

def main():
    # dataFilePath = ['data/ANightmareonElmStreet/data.npy', 'data/FinalFantasy/data.npy', 'data/WizardsAndWarriors/data.npy']
    # labelFilePath = ['data/ANightmareonElmStreet/label.npy', 'data/FinalFantasy/label.npy', 'data/WizardsAndWarriors/label.npy']
    dataFilePath = 'data/FinalFantasy/data.npy'
    labelFilePath = 'data/FinalFantasy/label.npy'
    testset_ratio = 0.15
    validset_ratio = 0.02

    data_manager = data_processing.Preprocessing(dataFilePath, labelFilePath, testset_ratio, validset_ratio)

    X_train, Y_train = data_manager.get_train_data()
    X_val, Y_val = data_manager.get_val_data()
    X_test, Y_test = data_manager.get_test_data()

    print(X_train.shape)
    print(Y_train.shape)

    # Example param for model
    model_params = utils.ParamDict(
        model = 'default',               # use 'default' to init default model
        input_shape = (256, 256, 3),     # (length, width, channel)
        lr = 5e-4,
        step_rate = 500,
        decay_rate = 0.96,
        num_epoch = 5,
        batch_size = 16,
        log_step = 50,
        num_training = X_train.shape[0],
        num_validation = X_val.shape[0],
        num_test = X_test.shape[0],
    )

    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        with tf.device('/cpu:0'):
            cur_model = model.FeatureExtractionModel(model_params)
            cur_model.train(sess, X_train, Y_train, X_val, Y_val)
            accuracy = cur_model.evaluate(sess, X_test, Y_test)
            print('***** test accuracy: %.3f' % accuracy)
            saver = tf.train.Saver()
            model_path = saver.save(sess, "tf_models/feature_extraction_v1.ckpt")
            print("Model saved in %s" % model_path)


if __name__ == "__main__":
    main()
