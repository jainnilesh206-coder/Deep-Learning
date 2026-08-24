# Deep Learning with Python

A structured learning repository covering Artificial Neural Networks (ANNs), model training, optimization, overfitting control, Convolutional Neural Networks (CNNs), and transfer learning using Python, TensorFlow/Keras, NumPy, Pandas, and scikit-learn.

> **Learning focus:** understand the theory, implement models from scratch, experiment with training choices, and deploy predictions through reusable Python workflows.

## Learning roadmap

1. Python and numerical foundations
2. Neural-network fundamentals
3. Regression with ANNs
4. Binary and multiclass classification
5. Model training and backpropagation
6. Optimizers and learning rates
7. Overfitting and regularization
8. CNNs for image classification
9. Transfer learning with pretrained CNNs
10. Deployment-oriented prediction code

## Python foundations

The notebooks use Python as the main implementation language for the complete deep-learning workflow.

### Core Python skills

- Variables, numeric types, strings, lists, tuples, dictionaries, and sets.
- Conditional statements, loops, comprehensions, and functions.
- Imports, modules, reusable code, and notebook-based experimentation.
- Exception handling, file paths, and command-line or interactive input.
- Object-oriented concepts and callback classes.
- Array-oriented programming with NumPy rather than repeated Python loops.

### NumPy

NumPy provides the numerical array foundation used by machine-learning libraries.

```python
import numpy as np

features = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 3.4, 5.4, 2.3],
])

print(features.shape)
print(features[:, 0])
```

Important practices:

- Keep features and labels in compatible two-dimensional or one-dimensional shapes.
- Use vectorized operations for efficient numerical computation.
- Check `shape`, `dtype`, and missing values before training.
- Use `np.argmax()` to convert multiclass probability outputs into a predicted class index.

### Pandas

Pandas is used to load, inspect, and prepare tabular datasets.

```python
import pandas as pd

data = pd.read_csv("Salary_Data.csv")
print(data.head())
print(data.info())
print(data.describe())
```

A typical preparation pattern is:

```python
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
```

Before modeling, inspect data types, null values, class balance, duplicate records, and feature ranges.

### scikit-learn preprocessing

The notebooks use scikit-learn for label encoding, feature scaling, and train-test splitting.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=10
)
```

Fit preprocessing objects only on training data in production workflows to avoid data leakage. Use the same fitted scaler and encoder during inference.

## Neural-network fundamentals

A neural network is a trainable function approximator inspired by biological neurons. A neuron calculates a weighted sum and bias, then applies an activation function:

\[
z = \sum_{i=1}^{n} w_i x_i + b
\]

\[
a = f(z)
\]

A multilayer perceptron (MLP), also called a fully connected ANN, contains:

- An input layer representing feature columns.
- One or more hidden layers that learn intermediate representations.
- An output layer that produces the prediction.

Activation functions introduce nonlinearity. Without them, stacking dense layers would still represent only a linear transformation.

## Output-layer design

Choose the output units and activation function according to the target variable.

| Problem | Output units | Activation | Typical loss |
|---|---:|---|---|
| Regression | 1 | Linear | Mean squared error or mean absolute error |
| Binary classification | 1 | Sigmoid | Binary cross-entropy |
| Multiclass classification with integer labels | Number of classes | Softmax | Sparse categorical cross-entropy |
| Multiclass classification with one-hot labels | Number of classes | Softmax | Categorical cross-entropy |

The linear activation is a passthrough: `f(z) = z`. Sigmoid returns a value between 0 and 1. Softmax converts class scores into a probability distribution whose values sum to 1.

## Hidden-layer activations

- **ReLU:** `max(0, z)`; a strong default for most deep networks.
- **Leaky ReLU:** preserves a small negative slope and can reduce inactive neurons.
- **Sigmoid:** useful in selected shallow architectures, but can saturate in deep networks.
- **Tanh:** outputs values between -1 and 1, but may also contribute to vanishing gradients.

Vanishing gradients occur when gradients become extremely small during backpropagation. Weight and bias updates then approach zero, slowing or stopping learning. ReLU-family activations are commonly preferred in hidden layers to reduce this problem.

## Training workflow

ANN training follows three connected stages:

1. **Forward propagation:** calculate weighted sums, activations, and predictions.
2. **Error calculation:** compare predictions with ground truth through a loss function.
3. **Backpropagation:** calculate gradients and update weights and biases.

Conceptually, gradient descent updates parameters in the direction that reduces loss:

\[
w_{new} = w - \eta \frac{\partial L}{\partial w}
\]

\[
b_{new} = b - \eta \frac{\partial L}{\partial b}
\]

Here, `η` is the learning rate and `L` is the loss.

A model is generally created, compiled, trained, evaluated, and then used for inference:

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1, activation="linear"),
])

model.compile(
    optimizer="sgd",
    loss="mean_squared_error",
    metrics=[tf.keras.metrics.R2Score()],
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
)
```

An epoch is one complete pass through the training data. A batch is the subset processed before a parameter update.

## Regression with ANN

The regression implementation uses a dense network to predict a continuous target, such as salary from years of experience.

Recommended workflow:

- Load the CSV with Pandas.
- Separate feature columns and the continuous target.
- Split the data into training and test sets.
- Scale features when appropriate.
- Use a single linear output unit.
- Compile with MSE or MAE.
- Track a regression metric such as R².
- Inspect both training and validation loss.

The uploaded regression notebook demonstrates a sequential model with dense hidden layers, stochastic gradient descent, MSE, validation data, and R² monitoring. Its long training run shows why epoch count, learning rate, and validation behavior should be monitored rather than selected blindly.

## Classification with ANN

### Binary classification

Binary labels should normally be represented as 0 and 1. Use one sigmoid output unit:

```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
```

### Multiclass classification

The Iris example contains 150 records, four numeric feature columns, and three balanced labels: Setosa, Versicolor, and Virginica. The workflow demonstrates label encoding, standardization, an 80/20 train-test split, a softmax output, and deployment-style inference.

For integer labels:

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y.ravel())

model = tf.keras.Sequential([
    tf.keras.Input(shape=(4,)),
    tf.keras.layers.Dense(12, activation="relu"),
    tf.keras.layers.Dense(12, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

For one-hot labels, use `tf.keras.utils.to_categorical()` and replace the loss with `categorical_crossentropy`.

During inference, scale the new record with the fitted scaler, obtain class probabilities, select the largest probability, and convert the index back to the original label:

```python
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
sample_scaled = scaler.transform(sample)
probabilities = model.predict(sample_scaled, verbose=0)
class_index = np.argmax(probabilities, axis=1)[0]
predicted_label = encoder.inverse_transform([class_index])[0]
```

## Optimizers and learning rate

An optimizer implements the parameter-update strategy used during backpropagation. The experiments compare or introduce:

- SGD.
- Adam.
- Nadam.
- RMSProp.
- AdaDelta.
- Lion.

Adam is a practical starting point for many problems, but the best choice depends on the dataset, architecture, loss landscape, and learning rate. Treat optimizer selection as an experiment, not a guarantee.

The learning rate controls the size of each update:

- Too large: training may oscillate or diverge.
- Too small: training may become extremely slow.
- Suitable: loss decreases steadily and validation performance improves.

Useful experiments include changing the optimizer, tuning the learning rate, adjusting SGD momentum, and comparing training and validation curves.

## Overfitting and regularization

Overfitting occurs when a model memorizes training examples but generalizes poorly to unseen data. Monitor the gap between training and validation metrics.

The uploaded examples and notes cover:

- Changing the optimizer and learning rate.
- Checking whether overfitting is acceptable for the domain problem.
- L1 and L2 regularization penalties.
- Dropout, which randomly disables units during training.
- Increasing the amount or quality of data.
- Changing the ANN architecture.
- Using validation-based stopping rules.

A practical Keras pattern is:

```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(10,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
```

Use early stopping with a patience value in real projects, and restore the best weights when validation performance is the main objective.

## CNNs for images

A CNN learns spatial features from images. A typical hierarchy progresses from edges and textures to shapes, object parts, and complete objects.

### Image preparation

Images should generally be:

- Resized to a consistent square shape.
- Stored with a consistent channel format.
- Normalized, commonly by dividing pixel intensities by 255.
- Organized into class-labelled directories or another clearly defined dataset format.

A color image is represented by height, width, and channels. RGB images have three channels. The uploaded cat-and-dog example uses 64 × 64 RGB images and directory-based generators.

### Convolution and pooling

A kernel or filter slides across an image and performs element-wise multiplication and summation to produce a feature map. Stride controls the movement of the filter. Padding adds a virtual border and helps preserve border information.

Pooling downsamples feature maps. Max pooling selects the largest value in a local region; average pooling calculates the local mean. Flattening converts the final feature maps into a one-dimensional vector before dense layers.

### CNN implementation

```python
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0
)

train_data = train_generator.flow_from_directory(
    "cats_and_dogs/train",
    target_size=(64, 64),
    batch_size=20,
    class_mode="binary",
)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(64, 64, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
```

For multiclass image classification, use a dense output with one unit per class, `softmax`, and the appropriate categorical cross-entropy loss.

## Transfer learning

Training a CNN from scratch requires the model to learn visual features from the beginning. Transfer learning reuses visual representations learned by a pretrained model, often trained on a large dataset such as ImageNet.

Two common approaches are:

1. Use the pretrained model as-is when its labels and task match the use case.
2. Use the pretrained convolutional stack as a base model, freeze its weights, replace the classifier head, and train the new head on a custom dataset.

Example pattern:

```python
base_model = tf.keras.applications.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(7, activation="softmax"),
])
```

The final dense layer has seven units because the custom task has seven classes. After the classifier head is trained, selected base layers may be unfrozen for careful fine-tuning with a small learning rate.

