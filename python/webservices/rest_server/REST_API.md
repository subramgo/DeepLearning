# Face Classification REST API
## v1
### path: /demographics/{model_id}/image/{image_id}
#### operation: POST

  * data: one image
  * returns: 
    * image ID given
    * gender classification results for each image ID

When new images are given:
1. they are assigned a image ID
2. they are added to the hdf5 file with all other training images ???
3. image is classified 
  * results stored in separate directory
  * classification result stored in a server database

#### operation: GET
Same as the above POST except assumes image has already been received and classified.


## v2

### path: /demographics/{model_id}/images
#### operation: POST
Same as `v1` path `/demographics/{model_id}/image/{image_id}` except for a batch. Image IDs are also part of the data payload.

#### operation: GET
Same as the above POST except assumes image has already been received and classified.

# Model Updates
### path: /demographics/model/{model_id}
This URL represents a particular model. model_id in the path can correspond to a `.h5` file. This serves as a customer ID as well as allows us to distinguish between different data domains e.g. one could be “starbucks faces” while another is "we put too many female faces in this one"

#### operation: POST

  * data: one or more images **with ground truth classes**

This is for updating the model with new labelled training data.

  1. Images are added to internal image database for this model
  2. Model is updated by training on new images
    - some minimum number of new images or just a single one before updating model?



