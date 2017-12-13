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
 * 0 - a single run on **raw** dataset, with predefined values of hyperparameters is evaluated,
 * 1 - **raw** dataset is selected, hyperparameters are sampled,
 * 2 - **features** dataset is selected, hyperparameters are sampled,
 * 3 - **HLfeatures** dataset is selected, hyperparameters are sampled, <br />
 
with *my_gpu_memo* in the range of 0 to 100:
 * a given percentage of GPU card memory is allocated for the script. In practice, less than 4GB is enough in this case. Therefore, if one has a 4GB card the *my_gpu_memo* could be set to 80 or 100, whereas for 16GB 20 or 25, <br />
 
with *my_hml*:  
 * a number telling the script how many loops with random search trials should be executed. If you wish to execute 100 trials and have 16GB card, you can run 4 scripts per 25 trials and *my_gpu_memo*=25.


##### Example:
In order to evaluate 16 trials on **raw** dataset, with randomly sampled hyperparameters on 0th GPU card, allocating 25% of its memory run:
```
CUDA_VISIBLE_DEVICES=0 python RNNrandHyp.py 1 25 16
```

---
### References
[1] [*"Analysis of Properties of an Active Linear Gesture Sensor"* K. Czuszynski, J. Ruminski, J. Wtorek](https://www.degruyter.com/downloadpdf/j/mms.2017.24.issue-4/mms-2017-0052/mms-2017-0052.pdf)  
[2] [*"Pose classification in the gesture recognition using the linear optical sensor"* K. Czuszynski, J. Ruminski, J. Wtorek](http://ieeexplore.ieee.org/document/8004989/)  
