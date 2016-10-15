package en.menghui.android.damp;

import Jama.Matrix;

public class Network {
	public Matrix input;
	public Matrix target;
	
	public Network(Matrix input, Matrix target) {
		this.input = input;
		this.target = target;
	}
	
	public void forwardProp() {
		FullyConnectedLayer fc1 = new FullyConnectedLayer(this.input.getColumnDimension(), 10, "sigmoid", 0.5);
		fc1.setInput(this.input, this.input, 60);
		SoftmaxLayer sf1 = new SoftmaxLayer(10, 4, 0.5);
		sf1.setInput(fc1.output, fc1.output, this.target, 60);
	}
	
	public void sgd() {
		
	}
}
