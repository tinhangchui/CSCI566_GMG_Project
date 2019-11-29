import argparse
import model
import utils
import data_processing
import tensorflow as tf
import numpy as np


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
    loss_name = "",
    model_name = "",
)


def train_model(model_name):
    pathPrefix = "C:\\Users\\ziang\\OneDrive\\work_space\\CSCI-566\\Project\\";
    dataFilePath = [
        # 'data\\a-nightmare-on-elm-street\\data.npy',
        # 'data\\argus-no-senshi-japan-\\data.npy',
        'data\\final-fantasy\\data.npy',
        # 'data\\super-mario-bros\\data.npy',
        # 'data\\wizards-and-warriors\\data.npy'
    ]

    labelFilePath = [
        # 'data\\a-nightmare-on-elm-street\\label.npy',
        #  'data\\argus-no-senshi-japan-\\label.npy',
         'data\\final-fantasy\\label.npy',
        #  'data\\super-mario-bros\\label.npy',
        #  'data\\wizards-and-warriors\\label.npy'
     ]
    for i in range(len(dataFilePath)):
        dataFilePath[i] = pathPrefix + dataFilePath[i];
    for i in range(len(labelFilePath)):
        labelFilePath[i] = pathPrefix + labelFilePath[i];
    # dataFilePath = 'data/FinalFantasy/data.npy'
    # labelFilePath = 'data/FinalFantasy/label.npy'

    testset_ratio = 0.15
    validset_ratio = 0.02

    print("loading data...")
    data_manager = data_processing.Preprocessing(dataFilePath, labelFilePath, testset_ratio, validset_ratio)

    print("splitting data...")
    X_train, Y_train = data_manager.get_train_data()
    X_val, Y_val = data_manager.get_val_data()
    X_test, Y_test = data_manager.get_test_data()

    print("data loading done!")

    MODEL_PARAMS.num_training = X_train.shape[0]
    MODEL_PARAMS.num_validation = X_val.shape[0]
    MODEL_PARAMS.num_test = X_test.shape[0]

    for loss_name in ["bpm_loss", "energy_loss", "tse_loss"]:
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            with tf.device('/gpu:0'):
                MODEL_PARAMS.loss_name = loss_name
                MODEL_PARAMS.model_name = model_name
                if MODEL_PARAMS.loss_name is "energy_loss":
                    cur_model = model.EnergyFeatureExtractionModel(MODEL_PARAMS)
                elif MODEL_PARAMS.loss_name is "tse_loss":
                    cur_model = model.TseFeatureExtractionModel(MODEL_PARAMS)
                else:
                    cur_model = model.BpmFeatureExtractionModel(MODEL_PARAMS)

                cur_model.train(sess, X_train, Y_train, X_val, Y_val)
                if MODEL_PARAMS.loss_name is not "bpm_loss":
                    accuracy = cur_model.evaluate(sess, X_test, Y_test)
                    print('***** test accuracy: %.3f' % accuracy)
                saver = tf.train.Saver()
                model_path = saver.save(sess, "tf_models/" + model_name + "/" + loss_name + "/" + model_name + "_" + loss_name + ".ckpt")
                print("Model saved in %s" % model_path)


def predict(model_path, model_name, data_path, num_prediction):
    tf.reset_default_graph()

    predicted_tse, predicted_bpm, predicted_energy = np.array([0]),np.array([0]),np.array([0])
    model_suffixes = ['_tse','_bpm','_energy']
    file_suffix = '_loss.ckpt'

    for model_suffix in model_suffixes:
        tf.compat.v1.reset_default_graph()
        sess = tf.Session()
        predict_data = data_processing.load_predict_data(data_path, num_prediction)
        if model_suffix is '_tse':
            predict_model = model.TseFeatureExtractionModel(MODEL_PARAMS)
            saver = tf.train.Saver()
            print(model_path + "/" + model_name + "/tse_loss/" + model_name + model_suffix + file_suffix)
            saver.restore(sess, model_path + "/" + model_name + "/tse_loss/" + model_name + model_suffix + file_suffix)
            predicted_tse = predict_model.predict(sess, predict_data)
        elif model_suffix is '_bpm':
            predict_model = model.BpmFeatureExtractionModel(MODEL_PARAMS)
            saver = tf.train.Saver()
            saver.restore(sess, model_path + "/" + model_name + "/bpm_loss/" + model_name + model_suffix + file_suffix)
            predicted_bpm = predict_model.predict(sess, predict_data)
        else:
            predict_model = model.EnergyFeatureExtractionModel(MODEL_PARAMS)
            saver = tf.train.Saver()
            saver.restore(sess, model_path + "/" + model_name + "/energy_loss/" + model_name + model_suffix + file_suffix)
            predicted_energy = predict_model.predict(sess, predict_data)

    predicted_labels = data_processing.merge_labels(predicted_tse, predicted_bpm, predicted_energy)
    np.savetxt('predict.out', predicted_labels)
    np.savetxt('predict_round.out', np.rint(predicted_labels), fmt='%d')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REPLACE WITH DESCRIPTION",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", "-m", type=str, help="model character")
    parser.add_argument("--predict", "-p", default=False, nargs='?', const=True, help="use train mode")
    parser.add_argument("--predict_file", "-f", type=str, default="tf_models/model_name",
                        help="path to model directory plus model prefix")
    parser.add_argument("--predict_data", "-d", type=str, help="model data for predict")
    parser.add_argument("--predict_num", "-n", default=10, help="how many data should be predicted in this data set")

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    if args.predict:
        assert args.predict_data is not None
        predict(args.predict_file, args.model_name, args.predict_data, int(args.predict_num))
    else:
        assert args.model_name
        train_model(args.model_name)
