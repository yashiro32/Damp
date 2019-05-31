package en.menghui.android.damp.networks;

import java.util.ArrayList;
import java.util.List;

import android.util.Log;
import Jama.Matrix;
import en.menghui.android.damp.layers.FullyConnectedLayer;
import en.menghui.android.damp.layers.Layer;
import en.menghui.android.damp.layers.SoftmaxLayer;

public class FeedForwardNetwork extends Network {
	private static final String TAG = "Feed Forward Network";

	public List<Layer> layers = new ArrayList<Layer>();
	
	public Matrix inputs;
	public Matrix targets;
	public int batchSize;
	
	public FeedForwardNetwork(Matrix inputs, Matrix targets, int batchSize) {
		this.name = "feed forward";
		this.inputs = inputs;
		this.targets = targets;
		this.batchSize = batchSize;
	}
	
	public void forwardProp() {
		/* FullyConnectedLayer fc1 = new FullyConnectedLayer(this.input.getColumnDimension(), 10, "sigmoid", 0.5);
		fc1.setInput(this.input, this.input, 60);
		SoftmaxLayer sf1 = new SoftmaxLayer(10, 4, 0.5);
		sf1.setInput(fc1.output, fc1.output, this.target, 60); */
		
		for (int i = 0; i < layers.size(); i++) {
			if (layers.get(i).type.equals("fully connected")) {
				if (i == 0) {
					((FullyConnectedLayer)layers.get(i)).forwardProp(this.inputs, this.inputs, batchSize);
				} else {
					((FullyConnectedLayer)layers.get(i)).forwardProp(layers.get(i-1).output, layers.get(i-1).output, batchSize);
				}
			} else if (layers.get(i).type.equals("softmax")) {
				((SoftmaxLayer)layers.get(i)).forwardProp(layers.get(i-1).output, layers.get(i-1).output, targets, batchSize);
			}
		}
	}
	
	public void backProp() {
		for (int i = layers.size()-1; i > -1; i--) {
			if (layers.get(i).type.equals("fully connected")) {
				/* if (i == layers.size()-1) {
					((FullyConnectedLayer)layers.get(i)).backProp(null);
				} else { */
					((FullyConnectedLayer)layers.get(i)).backProp(layers.get(i+1).bpOutput);
					((FullyConnectedLayer)layers.get(i)).optimize();
				// }
			} else if (layers.get(i).type.equals("softmax")) {
				((SoftmaxLayer)layers.get(i)).backProp(null);
				((SoftmaxLayer)layers.get(i)).optimize();
			}
		}
	}
	
	public void fit() {
		for (int i = 0; i < epochs; i++) {
			forwardProp();
			
			Log.i(TAG, "Accuracy: " + this.layers.get(this.layers.size()-1).evaluator.evaluate(this.layers.get(this.layers.size()-1).yOut, this.targets));
			
		    backProp();
		}
	}
	
	public void predict(Matrix set) {
		for (int i = 0; i < layers.size(); i++) {
			// Set the isTraining variable to false.
			layers.get(i).isTraining = false;
			
			if (layers.get(i).type.equals("fully connected")) {
				if (i == 0) {
					((FullyConnectedLayer)layers.get(i)).forwardProp(set, set, batchSize);
				} else {
					((FullyConnectedLayer)layers.get(i)).forwardProp(layers.get(i-1).output, layers.get(i-1).output, batchSize);
				}
			} else if (layers.get(i).type.equals("softmax")) {
				((SoftmaxLayer)layers.get(i)).forwardProp(layers.get(i-1).output, layers.get(i-1).output, targets, batchSize);
			}
		}
	}
	
	public void sgd() {
		
	}
}
