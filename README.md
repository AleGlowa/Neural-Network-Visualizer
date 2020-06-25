# **NN Visualizer**
The educational project to build an entire training process of a dense artificial neural network from scratch (not using any deep learning frameworks).
I'm planning to expand this project by visualizing training process on the website.

## **General info**
Until this moment I've implemented training simulation with Numpy.
My motivation was to fully understand backpropagation algorithm behind a dense ANN.

## **Example output**
```bash
python test.py
```
![Alt text](/NN/Resources/For_readme/training_sim.png?raw=true "Optional title")

## **How to test it to your needs**
- Set number of epochs and batch size
```python
NUM_EPOCHS = 10
BATCH_SIZE = 2
```

- Define inputs as 2darray where columns are training examples and rows are features for ANN
```python
inputs = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=T_PRECISION)  # 2 examples
```

- Define targets as 2darray where columns are targets and rows are output neurons for each example( ex. for binary problem it will be 1 output neuron )
```python
targets = np.array([[1, 0]])  # 2 targets
```

- Define a network structure as a list where its lenght is number of layers( along with input layer ) and
each number within the list indicate amount of neurons in each layer, 2 higher-order functions which initialize respectively weights and biases
( here weights are initialized with 0.01 and biases with 1 )
```python
# 3 hidden layers
dense_nn = NN_Dense([inputs.shape[0], 3, 2, 1], init.with_value(0.01), init.with_value(1))
```

- Define an activation function which will be applied in each hidden layer( here is used sigmoid )
NOTE: At this point, you can only use the sigmoid because I don't have any other activation functions derivatives defined which are required in the
backpropagation process!
```python
output = dense_nn.forward(inputs[:, id_batch * BATCH_SIZE : upper_bound], sigmoid)
```

- Define a loss function( here cross entropy )
```python
loss = cross_entropy(output, targets[:, id_batch * BATCH_SIZE : upper_bound])
```

- Define an optimazation higher-order function( here simple gradient descent with learning rate set to 0.02 )
```python
dense_nn.step(gradient_descent(lr=0.02))
```

## **Technologies**
- Python 3.7.7
- Numpy 1.18.1

## **Setup**
To install all dependencies, run the command `bash pip install -r requirements.txt`

## **Features**
Ready features:
- training simulation

To-do list( from the most important ):
- calculation of derivatives of the functions other than sigmoid
- support for input data in csv file format
- training visualization on the website

## **Status**
Project is: in progress

## **Inspiration**
Forward propagation and backpropagation based on [this video](https://www.youtube.com/watch?v=x_Eamf8MHwU)

## **Contact**
My email: alglowa1@gmail.com__
Linkedin: [Alex Głowacki](https://www.linkedin.com/in/alex-g%C5%82owacki-72941113a/)
