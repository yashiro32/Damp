package en.menghui.android.damp.examples;

import java.util.ArrayList;
import java.util.List;

import en.menghui.android.damp.R;
import en.menghui.android.damp.layers.FullyConnectedLayer;
import en.menghui.android.damp.layers.Layer;
import en.menghui.android.damp.layers.SoftmaxLayer;
import en.menghui.android.damp.networks.FeedForwardNetwork;
import en.menghui.android.damp.utils.NeuralNetUtils;
import Jama.Matrix;
import android.app.Activity;
import android.os.Bundle;
import android.util.Log;

public class TitanicLRExample extends Activity {
	private static final String TAG = "Titanic LR Example";
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		
		TitanicDataSet dataSet = new TitanicDataSet(this);
		int[] columnsToIgnore = {2, 7};
		dataSet.loadDataSet(0, columnsToIgnore);
		
		// NeuralNetUtils.printMatrix(dataSet.featuresMatrix.getMatrix(0, 0, 0, 5));
		
		// Build the neural network.
		FullyConnectedLayer fc1 = new FullyConnectedLayer(6, 32, "sigmoid", 0.0);
		fc1.learningRate = 0.01;
		// fc1.regLambda = 0.0;
		// fc1.useBatchNormalization = true;
		// fc1.learningRateDecayFactor = 0.1;
		// fc1.useLRDecay = true;
		// FullyConnectedLayer fc2 = new FullyConnectedLayer(32, 32, "sigmoid", 0.0);
		// fc2.learningRate = 0.1;
		// fc2.useBatchNormalization = true;
		// FullyConnectedLayer fc3 = new FullyConnectedLayer(32, 64, "sigmoid", 0.0);
		// fc3.learningRate = 0.01;
		// fc3.useBatchNormalization = true;
		SoftmaxLayer sf1 = new SoftmaxLayer(32, 2, 0.0);
		sf1.learningRate = 0.01;
		// sf1.regLambda = 0.0;
		// sf1.learningRateDecayFactor = 0.1;
		// sf1.useLRDecay = true;
		
		FeedForwardNetwork network = new FeedForwardNetwork(NeuralNetUtils.featureNormalize(dataSet.featuresMatrix), dataSet.labelsMatrix, 16);
		List<Layer> layers = new ArrayList<Layer>();
		layers.add(fc1);
		// layers.add(fc2);
		// layers.add(fc3);
		layers.add(sf1);
		network.layers = layers;
		network.epochs = 100;
		network.fit();
		
		NeuralNetUtils.printMatrix(network.layers.get(network.layers.size()-1).output);
		NeuralNetUtils.printMatrix(network.layers.get(network.layers.size()-1).yOut);
		
		List<String[]> testList = new ArrayList<String[]>();
		String[] dicaprio = {"0", "3", "JackDawson", "male", "19", "0", "0", "N/A", "5.0000"};
		String[] winslet = {"1", "1", "Rose DeWitt Bukater", "female", "17", "1", "2", "N/A", "100.0000"};
		testList.add(dicaprio);
		testList.add(winslet);
		
		TitanicDataSet testSet = preprocess(testList, 0, columnsToIgnore);
		network.predict(testSet.featuresMatrix);
		
		NeuralNetUtils.printMatrix(network.layers.get(network.layers.size()-1).output);
		NeuralNetUtils.printMatrix(network.layers.get(network.layers.size()-1).yOut);
		
		// oneHiddenLayerImplementation();
	}
	
	public TitanicDataSet preprocess(List<String[]> list, int labelColumn, int[] columnsToIgnore) {
		TitanicDataSet set = new TitanicDataSet(this);
		set.listToMatrix(list, labelColumn, columnsToIgnore);
		
		return set;
	}
	
	private void oneHiddenLayerImplementation() {
		TitanicDataSet dataSet = new TitanicDataSet(this);
		int[] columnsToIgnore = {2, 7};
		dataSet.loadDataSet(0, columnsToIgnore);
		
		// NeuralNetUtils.printMatrix(dataSet.featuresMatrix.getMatrix(0, 10, 0, 5));
		
		Matrix W1 = Matrix.random(6, 32);
		Matrix b1 = new Matrix(1, 32, 0.0);
		
		Matrix W2 = Matrix.random(32, 2);
		Matrix b2 = new Matrix(1, 2, 0.0);
		
		Matrix out = null;
		
		for (int j = 0; j < 100; j++) {
			// Forward propagation
			Matrix z1 = NeuralNetUtils.add(dataSet.featuresMatrix.times(W1), b1);
			Matrix a1 = NeuralNetUtils.sigmoid(z1, false);
			
			Matrix z2 = NeuralNetUtils.add(a1.times(W2), b2);
			Matrix exp = NeuralNetUtils.sigmoid(z2, false);
			
			out = exp;
			// NeuralNetUtils.printMatrix(exp);
			
			// Back Propagation
			// Matrix delta3 = y.minus(exp);
			Matrix delta3 = exp.copy();
			for (int k = 0; k < delta3.getRowDimension(); k++) {
				delta3.set(k, (int)dataSet.labelsMatrix.get(k, 0), delta3.get(k, (int)dataSet.labelsMatrix.get(k, 0)) - 1.0);
			}
			
			Matrix dW2 = a1.transpose().times(delta3);
			Matrix db2 = NeuralNetUtils.sum(delta3, 0);
			Matrix delta2 = delta3.times(W2.transpose()).arrayTimes(NeuralNetUtils.sigmoid(a1, true));
			
			Matrix dW1 = dataSet.featuresMatrix.transpose().times(delta2);
			Matrix db1 = NeuralNetUtils.sum(delta2, 0);
			
		    // Add regularization terms (b1 and b2 don't have regularization terms)
			dW2.plusEquals(W2.times(0.01));
			dW1.plusEquals(W1.times(0.01));
			
			W1.plusEquals(dW1.times(-0.001));
			b1.plusEquals(db1.times(-0.001));
			W2.plusEquals(dW2.times(-0.001));
			b2.plusEquals(db2.times(-0.001));
		
		}
		
		NeuralNetUtils.printMatrix(out);
	    NeuralNetUtils.printMatrix(NeuralNetUtils.argmax(out, 1));
		
	}
}
