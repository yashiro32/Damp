### DAMP

Damp is a Deep Learning Library for the Android Platform.

### Third party libraries

Uses JAMA a basic linear algebra package for Java.
(http://math.nist.gov/javanumerics/jama/)

### Example Code

##### Binary Classification using Titanic Dataset. 
##### Reference from Karthik M Swamy github code for TFLearn (https://github.com/karthikmswamy/TFTutorials/blob/master/TFLearn_Tutorials/Titanic_TFLearn.py).

```
TitanicDataSet dataSet = new TitanicDataSet(this);
int[] columnsToIgnore = {2, 7};
dataSet.loadDataSet(0, columnsToIgnore);

// Build the neural network.
FullyConnectedLayer fc1 = new FullyConnectedLayer(6, 32, "sigmoid", 0.0);
fc1.learningRate = 0.01;
fc1.regLambda = 0.0;
fc1.useBatchNormalization = true;
fc1.learningRateDecayFactor = 0.1;
fc1.useLRDecay = true;
fc1.activation = new SigmoidActivation();
fc1.optimizer = new SGDOptimizer(0.9, 0.01);
fc1.useDropout = true;
fc1.dropoutP = 0.6;
SoftmaxLayer sf1 = new SoftmaxLayer(32, 2, 0.0);
sf1.activation = new SigmoidActivation();
sf1.optimizer = new SGDOptimizer(0.9, 0.01);

FeedForwardNetwork network = new FeedForwardNetwork(NeuralNetUtils.featureNormalize(dataSet.featuresMatrix, 0), dataSet.labelsMatrix, 16);
List<Layer> layers = new ArrayList<Layer>();
layers.add(fc1);
layers.add(sf1);
network.layers = layers;
network.epochs = 100;
network.fit();

// Test Data.
List<String[]> testList = new ArrayList<String[]>();
String[] dicaprio = {"0", "3", "JackDawson", "male", "19", "0", "0", "N/A", "5.0000"};
String[] winslet = {"1", "1", "Rose DeWitt Bukater", "female", "17", "1", "2", "N/A", "100.0000"};
testList.add(dicaprio);
testList.add(winslet);

TitanicDataSet testSet = preprocess(testList, 0, columnsToIgnore);
network.predict(testSet.featuresMatrix);

NeuralNetUtils.printMatrix(network.layers.get(network.layers.size()-1).output);
NeuralNetUtils.printMatrix(network.layers.get(network.layers.size()-1).yOut);
```




