# western_match

__Resources for identifying duplicate gel images from biomedical research articles__

## Contents of the repo
### Notebooks:
Sample notebooks demonstrating the ability of the pipeline to:
- __Western\_Detection:__ Identify Western blot images (and indeed any gel images) from figures in biomedical research articles using an Inception V2 object detector implemented in TensorFlow.
- __Duplicate\_Detection:__ Compare images of Western blots to a pre-existing library of images (or to other individual images) to identify duplicates.

### exported_model:
The Inception V2 object detector, re-trained via Transfer Learning to identify Western blot images. This is the model used to extract Western blots from figures.

### hashed_westerns:
The Western blot image library used in [Duplicate_Detection](Notebooks/Duplicate_Detection.ipynb). See the notebook for an explanation of how this was generated.

### kmeans_models:
K-Means clustering models used to generate groups of related ORB and SIFT feature "words" for hashing gel images. See [Duplicate_Detection](Notebooks/Duplicate_Detection.ipynb) for details of how this works.

### Other:
- gcloud_ssd_inception_v2_gel_2.config: Config file for transfer learning with the Inception model. Provided for parameter reference.
