# ML_linear_regression_example
This is an example using linear regression for prediction in machine learning using the gradient descent algorithm.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Installing

This python script was developed using python 2.7.12, make sure the correct version is installed

Install the following python module using the following command line:

```
pip install os
pip install numpy
pip install pandas
pip install matplotlib

```

## Running the tests

To run the unit tests:

```
python validation_test.py
```

## Running the program:

To run the program:

Use the following command line:
```
python main.py
```

To edit the sample data file, edit the code directly in main.py in the following line:

```
path = os.getcwd() + '/sampledata1.txt'

```

## Gradient Descent Algorithm for Linear Regression

The objective of linear regression is to minimize the cost function:

![first equation](http://latex.codecogs.com/gif.latex?J%28%5CTheta%20%29%20%3D%20%5Cfrac%7B1%7D%7B2m%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%28h_%7B%5CTheta%20%7D%28x%5E%7B%28i%29%7D%29-y%5E%7B%28i%29%7D%29%5E%7B2%7D)

where the hypothesis h θ (x) is given by the linear model:

![second equation](http://latex.codecogs.com/gif.latex?h_%7B%5CTheta%20%7D%28x%29%3D%5CTheta%20%5E%7BT%7D%3D%5CTheta%20_%7B0%7D&plus;%5CTheta%20_%7B1%7Dx_%7B1%7D)

The θ j values are the values you will adjust to minimize cost J(θ). In batch gradient descent, each iteration performs the update:

![third equation](http://latex.codecogs.com/gif.latex?%5CTheta%20_%7Bj%7D%3D%5CTheta%20_%7Bj%7D-%5Calpha%20%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%28h_%7B%5CTheta%20%7D%28x%5E%7B%28i%29%7D%29-y%5E%7B%28i%29%7D%29x_%7Bj%7D%5E%7B%28i%29%7D)

Simultaneously updating all θ j for all j. With each step of gradient descent, your parameters θ j come closer to the  optimal values that will achieve the lowest cost J(θ).













