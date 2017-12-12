# RNNforGestures
This repo contains materials, source code and usage description of a script, which trains and evaluates Recurrent Neural Networks (RNNs) applied for hand gesture recognition. The input data comes from the optical linear sensor described in [1-2].

### Usage
The RNNrandHyp.py python script needs following packages to be installed:
```
sklearn, tensorflow, numpy, pandas, enum, datetime, sys
```

In order to run the script from console type:
```
python RNNrandHyp.py my_option my_gpu_memo my_hml
```
where with *my_option* equal to:
 * 0 - a single run on **raw** dataset, with predefined values of hyperparameters is evaluated
 * 1 - **raw** dataset is selected, hyperparameters are sampled
 * 2 - **features** dataset is selected, hyperparameters are sampled
 * 3 - **HLfeatures** dataset is selected, hyperparameters are sampled


---
### References
[1] [*"Analysis of Properties of an Active Linear Gesture Sensor"* K. Czuszynski, J. Ruminski, J. Wtorek](https://www.degruyter.com/downloadpdf/j/mms.2017.24.issue-4/mms-2017-0052/mms-2017-0052.pdf)  
[2] [*"Pose classification in the gesture recognition using the linear optical sensor"* K. Czuszynski, J. Ruminski, J. Wtorek](http://ieeexplore.ieee.org/document/8004989/)  
