from keras import layers
from keras import applications
import keras.backend as K
import numpy as np 
from keras.models import Model


K.set_image_data_format('channels_last')
np.random.seed(123)



def get_model():
	xception_base = applications.Xception(weights = None, include_top = False)
	

	anchor_input = layers.Input(shape=(200,200,3))
	pos_input     = layers.Input(shape=(200,200,3))
	neg_input     = layers.Input(shape=(200,200,3))

	anchor_features = xception_base(anchor_input)
	pos_features     = xception_base(pos_input)
	neg_features    = xception_base(neg_input)

	print(anchor_features.shape)
	merged_features = layers.concatenate([anchor_features, pos_features, neg_features], axis=-1)

	
	model = Model(inputs = [anchor_input, pos_input, neg_input], outputs = merged_features)

	print(model.summary())





if __name__ == "__main__":
	get_model()

