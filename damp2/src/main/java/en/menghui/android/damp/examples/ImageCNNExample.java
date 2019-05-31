package en.menghui.android.damp.examples;

import java.util.Arrays;

import en.menghui.android.damp.R;
import en.menghui.android.damp.activations.ReluActivation;
import en.menghui.android.damp.activations.SigmoidActivation;
import en.menghui.android.damp.activations.TanhActivation;
import en.menghui.android.damp.arrays.Tensor;
import en.menghui.android.damp.layers.ConvLayer;
import en.menghui.android.damp.layers.ConvolutionLayer2;
import en.menghui.android.damp.layers.FullyConnectedLayer;
import en.menghui.android.damp.layers.PoolingLayer2;
import en.menghui.android.damp.layers.SoftmaxLayer;
import en.menghui.android.damp.optimizations.AdamOptimizer;
import en.menghui.android.damp.optimizations.SGDOptimizer;
import en.menghui.android.damp.utils.NeuralNetUtils;
import en.menghui.android.damp.utils.Volume;
import android.app.Activity;
import android.os.Bundle;
import android.util.Log;

public class ImageCNNExample extends Activity {
	private static final String TAG = "Image CNN Example";
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		
		/* ConvLayer layer = new ConvLayer(5, 5, 8);
		layer.optimizer = new AdamOptimizer(0.9, 0.999, 0.01);
		layer.init(28, 28, 3);
		List<Volume> list = new ArrayList<Volume>();
		Volume volume = new Volume(28, 28, 3);
		list.add(volume);
		Volume volume2 = new Volume(28, 28, 3);
		list.add(volume2); 
		
		Log.d(TAG, "Training starts");
		Log.d(TAG, "Input length: " + list.get(0).weights.length);
		long timeStart = System.currentTimeMillis();
		List<Volume> out = layer.forwardProp(list);
		Log.d(TAG, "Output length: " + out.get(0).weights.length);
		layer.backProp();
		layer.optimize();
		Log.d(TAG, "Training ends at " + (System.currentTimeMillis() - timeStart)); */
		
		// Tensor imgs = new Tensor(Arrays.asList(1, 1, 8, 8), true);
		// Tensor filters = new Tensor(Arrays.asList(1, 1, 7, 7), true);
		// Tensor convout = new Tensor(Arrays.asList(1, 1, 8, 8));
		
		int epochs = 100;
		int miniBatchSize = 60;
		
		MnistDataSet dataSet = new MnistDataSet(this);
		int[] columnsToIgnore = {};
		dataSet.loadDataSet(0, columnsToIgnore, false);
		
		Tensor imgs = new Tensor(Arrays.asList(100, 1, 28, 28));
		imgs.tmat = dataSet.featuresMatrix;
		Tensor filters = new Tensor(Arrays.asList(1, 1, 5, 5), true);
		ConvolutionLayer2 cl1 = new ConvolutionLayer2(100, 1, 1, 1, 28, 28, 5, 5);
		cl1.filters = filters;
		cl1.useBatchNormalization = true;
		cl1.activation = new ReluActivation();
		cl1.optimizer = new SGDOptimizer(0.9, 0.01);
		
		PoolingLayer2 pl1 = new PoolingLayer2(100, 1, 28, 28, 2, 2, 2, 2);
		
		FullyConnectedLayer fc1 = new FullyConnectedLayer(196, 28, "sigmoid", 0.0);
		fc1.useBatchNormalization = true;
		fc1.activation = new TanhActivation();
		fc1.optimizer = new SGDOptimizer(0.9, 0.01);
		
		SoftmaxLayer sf1 = new SoftmaxLayer(28, 10, 0.0);
		sf1.activation = new SigmoidActivation();
		sf1.optimizer = new SGDOptimizer(0.9, 0.01);
		
		for (int i = 0; i < epochs; i++) {
			cl1.forwarProp(imgs, miniBatchSize);
			pl1.forwardProp(cl1.output, miniBatchSize);
			fc1.forwardProp(pl1.output.tmat, pl1.output.tmat, miniBatchSize);
			sf1.forwardProp(fc1.output, fc1.output, dataSet.labelsMatrix, miniBatchSize);
			
			Log.d(TAG, "Iteration: " + (i+1) + " Accuracy: " + sf1.evaluator.evaluate(sf1.yOut, dataSet.labelsMatrix));
			
			sf1.backProp(null);
			sf1.optimize();
			fc1.backProp(sf1.bpOutput);
			fc1.optimize();
			Tensor tensor = new Tensor(pl1.output.shape);
			tensor.tmat = fc1.bpOutput;
			pl1.backProp(tensor);
			cl1.backProp(pl1.bpOutput);
			cl1.optimize();
		}
		
		// NeuralNetUtils.printMatrix(sf1.yOut);
	}
}
