package en.menghui.android.damp;

import Jama.Matrix;

public class Layer {
	public String type = "";
	
	public Matrix W;
	public Matrix b;
	
	public Matrix dW;
	public Matrix db;
	
	public int nIn;
	public int nOut;
	
	public Matrix input;
	public Matrix output;
	
	public Matrix yOut;
	public Matrix target;
	
	public Matrix dropoutInput;
	public Matrix dropoutOutput;
	
	public Matrix bpInput;
	public Matrix bpOutput;
	
	public Matrix params;
	public Matrix grads;
	
	public String activationFunction;
	
	public double regLambda = 0.01;
	public double learningRate = 0.01;
	
	public double dropoutP;
	
	public void sgd() {
		// Add regularization terms (b1 and b2 don't have regularization terms)
		dW.plusEquals(this.W.times(regLambda));
		
		// Gradient descent parameter update
		this.W.plusEquals(dW.times(-learningRate));
		this.b.plusEquals(db.times(-learningRate));
	}
}
