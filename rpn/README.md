## Mask RCNN Training Repository (An Instance Segmentation Model)



## In Details
```
├──  dataset
│    └── images  - Store the Images here
│    └── masks - Store the Masks here
│    └── data.txt - It stores the relative path of the images & masks 
│
├── train.py
├── train.ipynb
├── test.ipynb - Custom Testings
```

## Steps

- Create the environment using the env.yaml file provided.
- Change the class ParcelDataset based on you required format. Currently we use two .png file, one for images and other for instance level mask images.
- Either use the train.ipynb or train.py file for training the model.

