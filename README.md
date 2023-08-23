# proper-ann
Simple and flexible Artificial Neural Network library for JavaScript/Node.js.

## Usage

### Importing

```js
const ProperANN = require('proper-ann');
```

### Instantiation

```js
let ann = new ProperANN({
  learningRate: .0001,
  layerNodeCounts: [100, 2000, 2000, 100],
  weightDecayFactor: .000001
});
```

The `layerNodeCounts` option represents the number of nodes at each layer starting with the input layer, hidden layers and the output layer.
There can be an arbitrary number of layers.

### Training

- The number of elements in the `inputs` and `targetOutputs` arrays (first and last arguments) must match the number of nodes in the input and output layer respectively (as specified in `layerNodeCounts` during instantiation).

- The elements in both `inputs` and `targetOutputs` arrays must be numbers (floating point numbers and negatives are supported).
The ideal range of numbers (e.g. when encoding strings) depends on the activation function. The default activation function is `ELU` which works well with small positive numbers between 0 and 5.

- `ELU` activation function is prone to exploding gradients, this can lead to `NaN` values in the outputs. In this case, you may want to reduce the range of numbers that you provide as input - Or you can use an alternative representation for your inputs and outputs; e.g. arrays of 1s and 0s as shown below.

```js
let { loss } = ann.train([1, 1], [0, 0]);
```

OR

```js
let { loss } = ann.trainBatch([
  [[1, 1], [0, 0]],
  [[1, 0], [0, 1]],
  [[0, 1], [1, 0]]
]);
```

### Inference

```js
let outputs = anna.run([1, 1]);
```

### Custom functions

Many functions which the ANN uses internally can be customized.
The default `activationFunction` is `ProperANN.eluActivationFunction` and the default `activationDerivativeFunction` (used to calculate gradients) is `ProperANN.eluActivationDerivativeFunction`.
The `ANN` class exposes a number of static functions which can be used by passing them to the ANN's constructor via the `activationFunction` and `activationDerivativeFunction` options during instantiation. It's important that the specified `activationFunction` and `activationDerivativeFunction` correspond.
