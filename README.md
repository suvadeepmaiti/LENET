## LENET-5 :</br>
</br>

### Problem Statement: </br>

Implementing Lenet-5, a simple and straighforward convolutional neural network architecture with gradient-based learning, for recognizing handwritten digits, from scratch. Lenet-5 comprises of 7 layers, not counting the input, all of which contains trainable parameters(weights). It takes an input of 32x32 pixel image and outputs the likelihood of the digit present in the image.

<img src="image/lenet.png" alt="Lenet-5" />

### Project objectives:</br>
• Build Lenet – 5 from scratch using basic libraries such as NumPy.</br>
• Evaluate with inbuilt Lenet – 5 model trained with libraries.</br>
• Achieve an accuracy close to the paper’s on MNSIT test set.</br>
### Deliverables:</br>

• Preprocess MNSIT Dataset containing 60000 samples.</br>
• Implementing hidden layers and forward pass.</br>
• Implementing back-propagation from scratch.</br>
• Visualizing and comparing the results on MNSIT data set.

</br>

### Code Structure
------------------

    ├── MNIST      #Dataset                       
    ├── images     #Images for readme file               
    ├── src  
    ├   ├── Results
    ├   ├   ├── #plot images      
    ├   ├── notebooks 
    ├   ├   ├── Lenet.ipynb
    ├   ├   ├── model-relu-tanh-28-512-adam.pickle  #Best model
    ├   ├   ├── #Some other experimental models
    ├   ├── scripts 
    ├   ├   ├── RBF_init.py
    ├   ├   ├── main.py
    ├   ├   ├── lenet.py
    ├   ├   ├── layers.py
    ├   ├── app 
    ├   ├   ├── static
    ├   ├   ├── templates
    ├   ├   ├── app.py
    ├   ├── MNIST_auto_Download.py                          
    └── README.md
-----------
</br>
In the above structure, the source code is found in the 'src' directory.The scripts folder in this directory consists of code related to layers, Lenet-5 and RBF weights inititialisation. The notebooks folder has Lenet.ipynb which has code for training, testing and analysis.
</br>
</br>

### Pre-requisites:
 
Before running the code, following python libraries are to be installed.

------------------
numpy  
matplotlib  
opencv2  
flask  
tabulate  
sklearn  
  
-----------
</br>

### Dataset:

We used the Modified NIST (MNIST) dataset which is a subset of the NIST database. It is a database of handwritten digits with a training set of 60,000 samples and a test set of 10,000 samples. </br>
Ref: http://yann.lecun.com/exdb/mnist/
</br>

### Running the drawing app:

We built an application to test the working of the Lenet-5 model. To check it please follow the below steps. </br>
    1. Go to the 'app' directory and then execute the 'app.py' file.</br> 
    2. From the terminal, you need to open the url (i.e 'http://127.0.0.1:5000/').</br>
    3. Draw the numbers from 0 to 9 in the given box and then click on predict. This will predict the written digit and the result is diplayed.
</br>

The result is supposed to be :</br>
</br>

<img src="image/lenet.gif" alt="Lenet-5" />

### References:</br>

Lenet :   Yann LeCun, Leon Bottou, Yoshua Bengio and Patrick HanerLenet, http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf,  IEEE NOVEMBER 1998

Adam : Diederik P. Kingma, Jimmy Lei Ba, https://arxiv.org/pdf/1412.6980.pdf,  ICLR 2015

Visual Tool : https://github.com/hugom1997/Flask-Cnn-Recognition-App
