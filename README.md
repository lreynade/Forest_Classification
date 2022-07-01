# Forest_Classification


This is a CodeAcademy project. This project is the culmination of the TensorFlow skill path.
Deep learning is used to predict forest cover type based only on cartographic variables. The
cover types are

* Spruce/Fir
* Lodgepole Pine
* Ponderosa Pine
* Cottonwood/Willow
* Aspen
* Douglas-fir
* Krummholz

The training and test sets contain 464809 and 116203, respectively. The model has two hidden layers.
From those layers and the output layer a total of 15,751 trainable parameters are obtained. The
following are hyperparameters that must be tuned.

* Learning_rate = 0.001
* Batch_size = 3000
* epochs = 50
* Validation_split = 0.12
* Dropout rate = 0.1

# Results

From training:

* CategoricalCrossEntropy: loss = 0.3458
* categorical_accuracy = 0.8572 
* accuracy = 0.9883
* val_loss = 0.3135
* val_cat_accuracy = 0.8737
* val_accuracy = 0.9905

From test test:

* CategoricalCrossEntropy: loss = 0.3085
* categorical_accuracy = 0.8766
* accuracy = 0.9908

Precision, recall, and F score from training and test results are comparable.

