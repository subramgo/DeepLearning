# Age & Gender Image Classification REST Service

## Structure 

  * REST interface
    * code modified from https://github.com/ansrivas/keras-rest-server
  * Predict by existing model
    * loaded from `model_path = '../models/age_gender_model_0_1.h5'`
  * [API](./REST_API.md)

## TODO
  * REST client to send image data
  * Process incoming image files
    * keras preprocessing
  * Return output


### POST


  * Provide one or more images
    1. return contains age and gender classification
    2. return has no data but images become training observations which can be retrieved by a later GET


