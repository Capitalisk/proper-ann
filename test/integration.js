const ProperANN = require('../index');

describe('Unit tests', function () {
  let words = [
    'construction',
    'cookies',
    'apple',
    'sugar',
    'simulation',
    'coordination'
  ];

  let testWords;
  let ann;

  beforeEach('Create test words', function () {
    testWords = [
      'constraction',
      'foostruction',
      'constructkon',
      'cinstructkon',
      'songtructkon',
      'constructio',
      'onstruction',
      'constractio',
      'constuction',
      'oookies',
      'caokiez',
      'sucar',
      'sugare',
      'sugars',
      'simulatio',
      'simulatton',
      'simulaton',
      'smulation',
      'coordinotion'
    ];

    ann = new ProperANN({
      learningRate: .0001,
      weightDecayFactor: .000001,
      layerNodeCounts: [12, 500, 600, 700, 800, 900, 1000, 1100, 1200, words.length],
    });
  });

  it('should be able to convert misspelled words into correct words', function () {
    console.log('Inference before training (mangled word -> original word):');
    for (let testWord of testWords) {
      let testWordArray = convertTextToNumberArray(testWord);
      let output = ann.run(testWordArray);
      console.log(testWord, convertProbArrayToWord(output));
    }

    let rounds = 500;
    console.log('----');
    console.log(`Starting ${rounds} training rounds...`);

    for (let i = 0; i < rounds; i++) {
      let curWord = words[i % words.length];
      let inputs = convertTextToNumberArray(mangleWord(curWord));
      let outputs = convertWordToProbArray(curWord);
      let { loss } = ann.train(inputs, outputs);

      console.log(i, 'LOSS:', loss);
    }

    console.log('----');
    console.log('Inference after training:');

    for (let testWord of testWords) {
      let testWordArray = convertTextToNumberArray(testWord);
      let output = ann.run(testWordArray);
      console.log(testWord, convertProbArrayToWord(output));
    }
  });

  // Convert a word into an array which represents a propability distribution of words.
  // Since, in this case, we know the exact word, we set its index to 1 and all others to 0.
  // For ANN outputs, there may be other values than 1 and 0.
  function convertWordToProbArray(word) {
    let index = words.indexOf(word);
    let probabilities = Array(words.length).fill(0);
    probabilities[index] = 1;
    return probabilities;
  }

  // Get the most likely word from a probability distribution.
  function convertProbArrayToWord(numberArray) {
    let maxNumber = Math.max.apply(Math, numberArray);
    let index = numberArray.indexOf(maxNumber);
    return words[index];
  }

  function convertTextToNumberArray(word, length = 12) {
    let diff = length - word.length;
    let paddedWord = word + (' ').repeat(Math.max(diff, 0));
    return paddedWord.split('').map(char => ((char === ' ' ? 123 : char.charCodeAt(0)) - 65) / 9.5 - 3);
  }

  function mangleWord(word, badCharError = .2, shiftError = .05, popError = .05) {
    let mangled = word.split('').map(
      char => Math.random() < badCharError ? String.fromCharCode(Math.floor(Math.random() * 73 + 49)) : char
    );
    if (Math.random() < shiftError) {
      mangled.shift();
    }
    if (Math.random() < popError) {
      mangled.pop();
    }
    return mangled.join('');
  }
});
