class ANN {
  constructor({
    layerNodeCounts,
    learningRate,
    weightDecayFactor,
    activationLeakFactor,
    activationFunction,
    activationDerivativeFunction,
    lossFunction,
    lossDerivativeFunction,
    weightDeltaFunction,
    biasDeltaFunction,
    errorDeltaFunction,
    learningRateFunction,
    weightInitFunction,
    errorInitFunction,
    biasInitFunction,
    trainInitFunction,
    trainCleanupFunction,
    layerComputationFunction,
    momentum,
    maxSafeOutputSize
  }) {
    this.maxWeightCount = 0;
    for (let i = 1; i < layerNodeCounts.length; i++) {
      let prevLayerNodes = layerNodeCounts[i - 1];
      let curLayerNodes = layerNodeCounts[i];
      this.maxWeightCount += prevLayerNodes * curLayerNodes;
    }
    if (learningRate == null) {
      learningRate = .0001;
    }
    if (maxSafeOutputSize == null) {
      maxSafeOutputSize = 2 ** 35;
    }
    if (weightDecayFactor == null) {
      weightDecayFactor = .000001;
    }
    if (activationLeakFactor == null) {
      activationLeakFactor = .01;
    }
    this.trainingSampleIndex = 0;
    this.layerNodeCounts = [...layerNodeCounts];
    this.layerCount = layerNodeCounts.length;
    this.weights = [];
    this.biases = [];
    this.momentum = momentum || .9;
    this.velocities = [];
    this.learningRate = learningRate;
    this.maxSafeOutputSize = maxSafeOutputSize;
    this.weightDecayFactor = weightDecayFactor;
    this.activationLeakFactor = activationLeakFactor;

    this.lossFunction = lossFunction || ANN.meanSquaredLossFunction;
    this.lossDerivativeFunction = lossDerivativeFunction || ANN.squaredLossDerivativeFunction;

    this.weightInitFunction = weightInitFunction || ANN.heWeightInitFunction;
    this.activationFunction = activationFunction || ANN.eluActivationFunction;
    this.activationDerivativeFunction = activationDerivativeFunction || ANN.eluActivationDerivativeFunction;

    this.errorInitFunction = errorInitFunction || ANN.simpleErrorInitFunction;

    this.biasInitFunction = biasInitFunction || ANN.zeroBiasInitFunction;
    this.layerComputationFunction = layerComputationFunction;

    this.errorDeltaFunction = errorDeltaFunction || ANN.simpleErrorDeltaFunction;
    this.weightDeltaFunction = weightDeltaFunction || ANN.velocityWeightDeltaFunction;
    this.biasDeltaFunction = biasDeltaFunction || ANN.simpleBiasDeltaFunction;
    this.learningRateFunction = learningRateFunction || ANN.simpleLearningRateFunction;

    this.trainInitFunction = trainInitFunction || ANN.simpleTrainInitFunction;
    this.trainCleanupFunction = trainCleanupFunction;

    this.weightCount = 0;

    let processingLayerCount = this.layerCount - 1;
    for (let i = 0; i < processingLayerCount; i++) {
      let curLayerNodeCount = layerNodeCounts[i];
      let nextLayerNodeCount = layerNodeCounts[i + 1];
      let layerWeights = [];
      let layerVelocities = [];
      for (let j = 0; j < curLayerNodeCount; j++) {
        let nodeWeights = [];
        let nodeVelocities = [];
        for (let k = 0; k < nextLayerNodeCount; k++) {
          let weightValue = this.weightInitFunction({i, j, k});
          if (weightValue != null) {
            nodeWeights.push(weightValue);
            nodeVelocities.push(0);
            this.weightCount++;
          }
        }
        layerWeights.push(nodeWeights);
        layerVelocities.push(nodeVelocities);
      }
      this.weights.push(layerWeights);
      this.velocities.push(layerVelocities);

      let layerBiases = [];
      for (let j = 0; j < nextLayerNodeCount; j++) {
        layerBiases.push(this.biasInitFunction({i, j}));
      }
      this.biases.push(layerBiases);
    }
  }

  static randomWeightInitFunction({i, j, k}) {
    return Math.random() - .5;
  }

  static heWeightInitFunction({i, j, k}) {
    return (Math.random() * 2 - 1) * Math.sqrt(2 / this.layerNodeCounts[i]);
  }

  static zeroBiasInitFunction({i, j}) {
    return 0;
  }

  static sigmoidActivationFunction(input) {
    return 1 / (1 + Math.exp(-input));
  }

  static sigmoidActivationDerivativeFunction({currentLayerInputs, currentLayerOutputs, j}) {
    let sigmoidOutput = currentLayerOutputs[j];
    return sigmoidOutput * (1 - sigmoidOutput);
  }

  static reluActivationFunction(input) {
    return Math.min(Math.max(input, 0), this.maxSafeOutputSize);
  }

  static reluActivationDerivativeFunction({currentLayerInputs, currentLayerOutputs, j}) {
    return currentLayerInputs[j] > 0 ? 1 : 0;
  }

  static leakyReluActivationFunction(input) {
    return Math.min(Math.max(input, input * this.activationLeakFactor), this.maxSafeOutputSize);
  }

  static leakyReluActivationDerivativeFunction({currentLayerInputs, currentLayerOutputs, j}) {
    return currentLayerInputs[j] > 0 ? 1 : this.activationLeakFactor;
  }

  static eluActivationFunction(input) {
    return Math.min(input > 0 ? input : 1 * (Math.exp(input) - 1), this.maxSafeOutputSize);
  }

  static eluActivationDerivativeFunction({currentLayerInputs, currentLayerOutputs, j}) {
    let input = currentLayerInputs[j];
    return input > 0 ? 1 : ANN.eluActivationFunction.call(this, input) + 1;
  }

  static geluActivationFunction(input) {
    return Math.min(0.5 * input * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (input + 0.044715 * input ** 3))), this.maxSafeOutputSize);
  }

  static geluActivationDerivativeFunction({currentLayerInputs, currentLayerOutputs, j}) {
    let input = currentLayerInputs[j];
    let cdf = 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (input + 0.044715 * input ** 3)));
    let pdf = Math.exp(-0.5 * input ** 2) / Math.sqrt(2 * Math.PI);
    return 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (input + 0.044715 * input ** 3))) + input * (1 - cdf) * pdf;
  }

  static meanSquaredLossFunction({outputs, targetOutputs}) {
    let totalLoss = 0;
    for (let i = 0; i < targetOutputs.length; i++) {
      totalLoss += (targetOutputs[i] - (outputs[i] || 0)) ** 2;
    }
    return totalLoss / targetOutputs.length;
  }

  static squaredLossDerivativeFunction({targetOutputs, currentLayerInputs, currentLayerOutputs, j}) {
    return 2 * (targetOutputs[j] - currentLayerOutputs[j]);
  }

  static simpleTrainInitFunction() {
    this.activationDerivatives = [];
    for (let i = 0; i < this.layerNodeCounts.length; i++) {
      this.activationDerivatives[i] = [];
    }
  }

  static simpleErrorInitFunction({targetOutputs, finalLayerInputs, finalLayerOutputs, i, j}) {
    let lossDerivative = this.lossDerivativeFunction({targetOutputs, currentLayerInputs: finalLayerInputs, currentLayerOutputs: finalLayerOutputs, j});
    let activationDerivative = this.activationDerivativeFunction({currentLayerInputs: finalLayerInputs, currentLayerOutputs: finalLayerOutputs, j});
    this.activationDerivatives[i][j] = activationDerivative;
    return lossDerivative * activationDerivative;
  }

  static simpleErrorDeltaFunction({learningRate, layerInputs, layerOutputs, nextLayerErrors, currentWeight, i, j, k}) {
    if (!this.activationDerivatives[i][j]) {
      this.activationDerivatives[i][j] = this.activationDerivativeFunction({currentLayerInputs: layerInputs[i], currentLayerOutputs: layerOutputs[i], j});
    }
    let activationDerivative = this.activationDerivatives[i][j];
    return nextLayerErrors[k] * currentWeight * activationDerivative;
  }

  static simpleWeightDeltaFunction({learningRate, layerInputs, layerOutputs, nextLayerErrors, i, j, k}) {
    let gradient = nextLayerErrors[k] * layerOutputs[i][j];
    let weightDecay = this.weightDecayFactor * this.weights[i][j][k];
    return learningRate * (gradient - weightDecay);
  }

  static velocityWeightDeltaFunction({learningRate, layerInputs, layerOutputs, nextLayerErrors, i, j, k}) {
    let gradient = nextLayerErrors[k] * layerOutputs[i][j];
    let weightDecay = this.weightDecayFactor * this.weights[i][j][k];
    this.velocities[i][j][k] = learningRate * (gradient - weightDecay) + this.momentum * this.velocities[i][j][k];

    return this.velocities[i][j][k];
  }

  static simpleBiasDeltaFunction({learningRate, layerOutputs, currentNodeError, nextLayerErrors, i, j}) {
    return learningRate * currentNodeError;
  }

  static simpleLearningRateFunction({learningRate, trainingSampleIndex, i, j}) {
    return learningRate;
  }

  propagateForward(inputs) {
    let layerOutputs = [[...inputs]];
    let layerInputs = [[...inputs]];
    let nodeLayerCount = this.layerNodeCounts.length;
    for (let i = 0; i < nodeLayerCount - 1; i++) {
      let curLayerNodeOutputs = layerOutputs[i];
      let curLayerNodeWeights = this.weights[i];

      let curLayerNodeCount = this.layerNodeCounts[i];
      let nextLayerNodeCount = this.layerNodeCounts[i + 1];
      let nextLayerWeightedSums = Array(nextLayerNodeCount).fill(0);

      for (let j = 0; j < curLayerNodeCount; j++) {
        for (let k = 0; k < nextLayerNodeCount; k++) {
          nextLayerWeightedSums[k] += curLayerNodeOutputs[j] * (curLayerNodeWeights[j][k] || 0);
        }
      }

      let nextLayerInputs = [];
      let nextLayerOutputs = [];
      for (let k = 0; k < nextLayerWeightedSums.length; k++) {
        let weightedSum = nextLayerWeightedSums[k];
        let nodeInput = weightedSum + this.biases[i][k];
        nextLayerInputs.push(nodeInput);
        nextLayerOutputs.push(this.activationFunction(nodeInput));
      }
      layerInputs.push(nextLayerInputs);
      layerOutputs.push(nextLayerOutputs);
    }
    return {
      layerInputs,
      layerOutputs
    };
  }

  run(inputs) {
    let { layerOutputs } = this.propagateForward(inputs);
    return layerOutputs[layerOutputs.length - 1];
  }

  trainSample(inputs, targetOutputs) {
    this.trainInitFunction({inputs, targetOutputs});
    let { layerInputs, layerOutputs } = this.propagateForward(inputs);
    let finalLayerInputs = layerInputs[layerInputs.length - 1];
    let finalLayerOutputs = layerOutputs[layerOutputs.length - 1];
    let finalLayerOutputCount = finalLayerOutputs.length;
    if (targetOutputs.length !== finalLayerOutputCount) {
      throw new Error(
        `Output array length of ${
          targetOutputs.length
        } did not match the ANN output node count of ${
          finalLayerOutputCount
        }`
      );
    }

    let nodeLayerCount = this.layerNodeCounts.length;

    let errors = [];
    let curLayerErrors = [];

    for (let j = 0; j < finalLayerOutputCount; j++) {
      let error = this.errorInitFunction({targetOutputs, finalLayerInputs, finalLayerOutputs, i: nodeLayerCount - 1, j});
      curLayerErrors.push(error);
    }
    errors.push(curLayerErrors);

    let weightDeltas = [];
    let biasDeltas = [];
    for (let i = nodeLayerCount - 2; i >= 0 ; i--) {
      weightDeltas[i] = [];
      if (i > 0) {
        biasDeltas[i - 1] = [];
      }
      let curLayerNodeCount = this.layerNodeCounts[i];
      let nextLayerNodeCount = this.layerNodeCounts[i + 1];
      curLayerErrors = [];
      let nextLayerErrors = errors[errors.length - 1];

      if (this.layerComputationFunction) {
        this.layerComputationFunction({
          learningRate: this.learningRate,
          layerInputs,
          layerOutputs,
          nextLayerErrors,
          currentLayerWeights: this.weights[i],
          i,
        });
      }

      for (let j = 0; j < curLayerNodeCount; j++) {
        weightDeltas[i][j] = [];
        let normalizedLearningRate = this.learningRateFunction({
          learningRate: this.learningRate,
          trainingSampleIndex: this.trainingSampleIndex,
          i,
          j
        });
        let error = 0;
        for (let k = 0; k < nextLayerNodeCount; k++) {
          if (this.weights[i][j][k]) {
            // Cannot calculate the error for input nodes
            if (i > 0) {
              error += this.errorDeltaFunction({
                learningRate: normalizedLearningRate,
                layerInputs,
                layerOutputs,
                nextLayerErrors,
                currentWeight: this.weights[i][j][k],
                i,
                j,
                k
              });
            }
            weightDeltas[i][j][k] = this.weightDeltaFunction({
              learningRate: normalizedLearningRate,
              layerInputs,
              layerOutputs,
              nextLayerErrors,
              i,
              j,
              k
            });
          }
        }
        // Input layer does not have biases.
        if (i > 0) {
          biasDeltas[i - 1][j] = this.biasDeltaFunction({
            learningRate: normalizedLearningRate,
            layerInputs,
            layerOutputs,
            currentNodeError: error,
            nextLayerErrors,
            i,
            j
          });
        }
        curLayerErrors.push(error);
      }
      errors.push(curLayerErrors);
    }

    let outputs = finalLayerOutputs;
    let result = {
      outputs,
      weightDeltas,
      biasDeltas,
      loss: this.lossFunction({targetOutputs, outputs})
    };

    if (this.trainCleanupFunction) {
      this.trainCleanupFunction(result);
    }

    return result;
  }

  applyWeightDeltas(weightDeltas) {
    for (let i = 0; i < weightDeltas.length; i++) {
      for (let j = 0; j < weightDeltas[i].length; j++) {
        for (let k = 0; k < weightDeltas[i][j].length; k++) {
          this.weights[i][j][k] += weightDeltas[i][j][k] || 0;
        }
      }
    }
  }

  applyBiasDeltas(biasDeltas) {
    for (let i = 0; i < biasDeltas.length; i++) {
      for (let j = 0; j < biasDeltas[i].length; j++) {
        this.biases[i][j] += biasDeltas[i][j];
      }
    }
  }

  applyWeightDeltasBatch(weightDeltas, batchSize) {
    for (let i = 0; i < weightDeltas.length; i++) {
      for (let j = 0; j < weightDeltas[i].length; j++) {
        for (let k = 0; k < weightDeltas[i][j].length; k++) {
          this.weights[i][j][k] += (weightDeltas[i][j][k] || 0) / batchSize;
        }
      }
    }
  }

  applyBiasDeltasBatch(biasDeltas, batchSize) {
    for (let i = 0; i < biasDeltas.length; i++) {
      for (let j = 0; j < biasDeltas[i].length; j++) {
        this.biases[i][j] += biasDeltas[i][j] / batchSize;
      }
    }
  }

  train(inputs, targetOutputs) {
    let { outputs, weightDeltas, biasDeltas, loss } = this.trainSample(inputs, targetOutputs);
    this.applyWeightDeltas(weightDeltas);
    this.applyBiasDeltas(biasDeltas);
    this.trainingSampleIndex++;
    return {
      outputs,
      loss
    };
  }

  trainBatch(trainingList) {
    let outputsList = [];
    let totalWeightDeltas = [];
    let totalBiasDeltas = [];
    let totalLoss = 0;
    for (let h = 0; h < trainingList.length; h++) {
      let { outputs, weightDeltas, biasDeltas, loss } = this.trainSample(trainingList[h][0], trainingList[h][1]);
      totalLoss += loss;
      outputsList.push(outputs);
      if (h === 0) {
        totalWeightDeltas = weightDeltas;
        totalBiasDeltas = biasDeltas;
      } else {
        for (let i = 0; i < weightDeltas.length; i++) {
          for (let j = 0; j < weightDeltas[i].length; j++) {
            for (let k = 0; k < weightDeltas[i][j].length; k++) {
              totalWeightDeltas[i][j][k] += weightDeltas[i][j][k] || 0;
            }
          }
        }
        for (let i = 0; i < biasDeltas.length; i++) {
          for (let j = 0; j < biasDeltas[i].length; j++) {
            totalBiasDeltas[i][j] += biasDeltas[i][j];
          }
        }
      }
    }
    this.applyWeightDeltasBatch(totalWeightDeltas, trainingList.length);
    this.applyBiasDeltasBatch(totalBiasDeltas, trainingList.length);
    this.trainingSampleIndex += trainingList.length;
    return {
      outputsList,
      loss: totalLoss / trainingList.length
    };
  }

  getWeights() {
    return this.weights.map(layerWeights => layerWeights.map(nodeWeights => ([...nodeWeights])));
  }

  getBiases() {
    return this.biases.map(layerBiases => ([...layerBiases]));
  }
}

module.exports = {
  ANN,
};
