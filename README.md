# RNNforGestures
Materials and source code utilized for gesture recognition employing the optical linear sensor and Recurrent Neural Networks (RNNs)

### Usage
The RNNrandHyp.py python script needs following packages to be installed:
```
sklearn, tensorflow, numpy, pandas, enum, datetime, sys
```

In order to run the script from the console type:
```
python RNNrandHyp.py my_option my_gpu_memo my_hml
```
where with *my_option* equal to:
 * 0 - a single run on **raw** dataset, with predefined values of hyperparameters is evaluated
 * 1 - **raw** dataset is selected, hyperparameters are sampled
 * 2 - **features** dataset is selected, hyperparameters are sampled
 * 3 - **HLfeatures** dataset is selected, hyperparameters are sampled
