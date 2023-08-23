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
let outputs = ann.run([1, 1]);
```

### Custom functions

Many functions which the ANN uses internally can be customized.
The default `activationFunction` is `ProperANN.eluActivationFunction` and the default `activationDerivativeFunction` (used to calculate gradients) is `ProperANN.eluActivationDerivativeFunction`.
The `ProperANN` class exposes a number of static functions which can be used by passing them to the ANN's constructor via the `activationFunction` and `activationDerivativeFunction` options during instantiation. It's important that the specified `activationFunction` and `activationDerivativeFunction` correspond.

### Serializing to and from model files

You can use the `proper-ann-serializer` module to serialize your trained ANN model to and from the file system like this:

```js
const ProperANNSerializer = require('proper-ann-serializer');

// ...

(async () => {
  let annSerializer = new ProperANNSerializer();

  try {
    // This will look for a directory called my-model.
    await annSerializer.loadFromDir(ann, 'my-model');
  } catch (error) {
    console.log('Could not find an existing ANN model, will start from scratch...');
  }

  // ...

  // Will throw if it fails to save.
  await annSerializer.saveToDir(ann, 'my-model');
})();
```

### License

### AGPL by default

By default, this product is issued under `AGPL-3.0`.

### MIT license for CLSK token holders

If you own `10K CLSK` tokens (https://capitalisk.com/), then you are subject to the less restrictive `MIT` license and are therefore exempt from the AGPL requirement of making the code of your derived projects public. This alternative license applies automatically from the moment that you acquire 10K or more CLSK tokens and it is valid so long as you continue to hold that amount of tokens.

If your CLSK balance falls below 10K, then you will be once again bound to the conditions of AGPL-3.0 after a grace period of 90 days; after this grace period, your derived project's code should be made public. Timestamps which can be used to prove ownership of CLSK tokens over time are recorded on the Capitalisk blockchain in a decentralized, immutable way so it is important that you hold 10K CLSK throughout your derived project's entire commercial life if you intend to keep the code private.

This exemption also applies to companies; in this case, the total CLSK holdings of the company plus those of its directors and board members must be greater than 10K multiplied by the maximum number of contributors which have worked on the project concurrently since the start of the project (e.g. according to records in the project's code repository). If a company falls out of compliance, the standard 90-day grace period applies before reverting back to the AGPL-3.0 license.

The amount of CLSK tokens which need to be held to qualify for the MIT license (and exemption from AGPL-3.0) may be revised downwards in the future but never upwards.
