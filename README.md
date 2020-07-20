# [A deep learning approach to Road Segmentation](https://github.com/marceloqueiros/DL-road-segmentation)

## Introduction
Automatic segmentation of roads can be useful for identifying roads from satellite images. In some countries, total mapping of roads does not exist and therefore this software proves to be of great importance.
Deep learning is part of a broader family of machine learning methods based on learning data representations, as opposed to task-specific algorithms. At the moment, Deep Learning is the state of the art computer vision method to solve complex problems without well-defined algorithmic rules. The identification of a road does not present trivial and well-defined rules.

[A deep learning approach to Road Segmentation](https://github.com/marceloqueiros/DL-road-segmentation) was implemented in a supervised way, meaning that labelled images were used (original image + road mask). The images used for training were taken from https://www.cs.toronto.edu/~vmnih/data/.

___
## Network
The network used was already known to succeeded in segmentation frameworks such as "unet", which presents the following architecture:

| <img src="Images_Readme/unet.jpg" alt="unet" width="400"> |
|:--:| 
| **Fig.1:** Unet - network used in this project. |

___
## Results
In addition to presenting the code for pre-processing and training the network, there is also a file that tests images, as default, in "Images/" folder. You should call this file through cmd in the form of "python test_images.py". Some optional parameters can be added like "image directory", "path to template", "width", "height" and "delay". If no parameters are given, the file will use the default parameters.
After that, the result will be like the following (real results):

| <img src="Images_Readme/1.png" alt="prediction" width="400"> |
|:--:| 
| **Fig.2:** Original / GrouthTruth / Predicted / Original + Predicted. |

| <img src="Images_Readme/2.png" alt="prediction" width="400"> |
|:--:| 
| **Fig.3:** Original / GrouthTruth / Predicted / Original + Predicted. |

| <img src="Images_Readme/3.png" alt="prediction" width="400"> |
|:--:| 
| **Fig.4:** Original / GrouthTruth / Predicted / Original + Predicted. |

| <img src="Images_Readme/4.png" alt="prediction" width="400"> |
|:--:| 
| **Fig.5:** Original / GrouthTruth / Predicted / Original + Predicted. |

___

## Windows Installation
Procedures to install necessary frameworks on Windows

### Download and install Python 3.6.x x64 version (confirm Pip option). 
[Python 3.6](https://www.python.org/downloads/release/python-362/)


### GPU configuration (Optional: Only required for TensorFlow GPU installation):  
   * Confirm compatible versions of _TensorFlow, Python, Compiler, CuDNN and CUDA Toolkit_.  
   [Versions](https://www.tensorflow.org/install/install_sources#tested_source_configurations)

   * Download and install **NVIDIA CUDA &reg; Toolkit**.  
   Make sure that the version is compatible.  
   [NVIDIA CUDA Toolkit](https://developer.NVIDIA.com/cuda-downloads)
   
   * Download **CuDNN**.  
   Make sure that the version is compatible.  
   [NVIDIA cuDNN](https://developer.NVIDIA.com/cudnn)

   * After installing the NVIDIA Toolkit, copy CuDNN files do NVIDIA CUDA Toolkit directory.  
   By default, it is located at _C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/vX.0_  
   (_**NOTE:** X represents the version of the CUDA Toolkit_)

     * Copy _cudnn/bin/cudnn64_X.dll_ to _C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/vX.0/bin/_
     
     * Copy _cudnn/include/cudnn.h_ to _C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/vX.0/include/_
     
     * Copy _cudnn/lib/x64/cudnn.lib_ to _C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/vX.0/lib/_


### Install TensorFlow
  * ```pip install --upgrade tensorflow-gpu (for the latest GPU version)```
    
  * _(Alternative)_ ```pip install tensorflow-gpu==1.6.0```
    
  * _(Alternative)_ ```pip install tensorflow (for the latest CPU version)```
    
  * _(Alternative)_ ```pip install tensorflow==1.6.0 (for a specific CPU version)```
    
  * _(Alternative)_ Direct download from [Windows Python's Libs](https://www.lfd.uci.edu/~gohlke/pythonlibs/#tensorflow).  
  After download, go to the directory where it was downloaded and make  
  ```pip install tensorflow‑X.X.0‑cp36‑cp36m‑win_amd64.whl```  
  (_**NOTE:** X is the version of the TensorFlow_)

  * #### Tensorflow Dependencies
    * Make sure that Visual C++ Redistributable 2015 (or 2017) x64 is installed.  
    [MS Visual C++ Redistributable 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48145)

### Install Keras
```pip install keras```


### Additional libraries dependencies
```pip install -r requirements.txt```

___
## Test Installation

Open a command's line and type  
```python -c "import tensorflow as tf; print(tf.__version__)```

