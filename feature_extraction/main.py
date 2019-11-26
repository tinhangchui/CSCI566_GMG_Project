import argparse
import model
import utils
import data_processing
import tensorflow as tf


# Example param for model
MODEL_PARAMS = utils.ParamDict(
    model = 'default',               # use 'default' to init default model
    input_shape = (256, 256, 3),     # (length, width, channel)
    output_dimension = 3,
    lr = 5e-4,
    step_rate = 500,
    decay_rate = 0.96,
    num_epoch = 5,
    batch_size = 16,                 # maybe too small now, enlarge this number if the dataset is large enough.
    log_step = 50,
    num_training = 0,                # will be changed automatically
    num_validation = 0,              # will be changed automatically
    num_test = 0,                    # will be changed automatically
)


def train_model():
    # dataFilePath = ['data/ANightmareonElmStreet/data.npy', 'data/FinalFantasy/data.npy', 'data/WizardsAndWarriors/data.npy']
    # labelFilePath = ['data/ANightmareonElmStreet/label.npy', 'data/FinalFantasy/label.npy', 'data/WizardsAndWarriors/label.npy']
    dataFilePath = ['data/FinalFantasy/data.npy', 'data/WizardsAndWarriors/data.npy']
    labelFilePath = ['data/FinalFantasy/label.npy', 'data/WizardsAndWarriors/label.npy']
    testset_ratio = 0.15
    validset_ratio = 0.02

    data_manager = data_processing.Preprocessing(dataFilePath, labelFilePath, testset_ratio, validset_ratio)

    X_train, Y_train = data_manager.get_train_data()
    X_val, Y_val = data_manager.get_val_data()
    X_test, Y_test = data_manager.get_test_data()

    MODEL_PARAMS.num_training = X_train.shape[0]
    MODEL_PARAMS.num_validation = X_val.shape[0]
    MODEL_PARAMS.num_test = X_test.shape[0]

    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        with tf.device('/cpu:0'):
            cur_model = model.FeatureExtractionModel(MODEL_PARAMS)
            cur_model.train(sess, X_train, Y_train, X_val, Y_val)
            accuracy = cur_model.evaluate(sess, X_test, Y_test)
            print('***** test accuracy: %.3f' % accuracy)
            saver = tf.train.Saver()
            model_path = saver.save(sess, "tf_models/feature_extraction_v1.ckpt")
            print("Model saved in %s" % model_path)


def predict(model_path, data_path, num_prediction):
    tf.reset_default_graph()

    predict_model = model.FeatureExtractionModel(MODEL_PARAMS)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    predict_data = data_processing.load_predict_data(data_path, num_prediction)
    predict_model.predict(sess, predict_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REPLACE WITH DESCRIPTION",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--predict", "-p", default=False, nargs='?', const=True, help="use train mode")
    parser.add_argument("--predict_file", "-f", type=str, default="tf_models/feature_extraction_v1.ckpt", help="model file for predict")
    parser.add_argument("--predict_data", "-d", type=str, help="model data for predict")
    parser.add_argument("--predict_num", "-n", default=10, help="how many data should be predicted in this data set")

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    if args.predict:
        assert args.predict_data is not None
        predict(args.predict_file, args.predict_data, args.predict_num)
    else:
        train_model()
