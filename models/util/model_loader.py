def load_model(model_id, dataset_name, restore_weight=True, image_shape=(320, 240, 3)):
    WEIGHT_FILES = '../weights/model_{:03d}'.format(model_id)
