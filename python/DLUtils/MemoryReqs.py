import numpy as np
from keras import backend as K


def model_memory_params(batch_size, model):
    shapes_count = int(np.sum([np.prod(np.array([s if isinstance(s, int) else 1 for s in l.output_shape])) for l in model.layers]))
    memory = shapes_count * 4 * batch_size
    memory_gb = np.round(memory / (1024 ** 3), 3)

    print ("Memory {} GB".format(memory_gb))
    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    print("Trainable weights {}".format(trainable_count))
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])
    print("Non Trainable weights {}".format(non_trainable_count))

    print("Total Parameters {}".format(trainable_count + non_trainable_count))






def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    layers = 1
    for l in model.layers:
        print("Layer {}".format(layers))
        layers+=1
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        print("Shape {} Memory {}".format(l.output_shape, single_layer_mem))

        shapes_mem_count += single_layer_mem

        print("Total {}".format(shapes_mem_count))

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    print("Trainable weights {}".format(trainable_count))
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])
    print("Non Trainable weights {}".format(non_trainable_count))

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


