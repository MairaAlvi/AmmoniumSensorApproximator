
# Cost Effective Soft Sensing in Wastewater Treatment Facilities
The developed code was tested over Ubuntu-18.04 with the following versions of packages. 

### Software Requirements

- Python 3.8.11
- TensorFlow 2.4.1
- Keras 2.4.0
- sklearn 0.24.1
- pandas 1.2.3
- matplotlib 3.4.1
- numpy 1.19.5

### Data

The HRAP Data set that we used in our paper is available in data folder.

### Trained Models

The trained model weights are available in ```trainedModels``` directory.

Once the enviornment has been set, the model can be trained by running ```GRUConv.py``` script as  

```py
  python GRUconv.py 
``` 

If you want to test the trained models, please check the code available in ```LoadTrainedModels.py```. 
