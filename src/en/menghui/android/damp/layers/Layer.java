package en.menghui.android.damp.layers;

import en.menghui.android.damp.optimizations.Optimizer;
import Jama.Matrix;

public class Layer {
	public String name = "";
	public String type = "";
	
	public Matrix W;
	public Matrix b;
	
	public Matrix dW;
	public Matrix db;
	
	// Memory variables for Ada, Adam, Ada Delta optimizer.
	public Matrix mW;
	public Matrix mb;
	
	// Memory variables for Adam, Ada Delta optimizer.
	public Matrix vW;
	public Matrix vb;
	
	// Hyper parameters for Adam optimizer.
	public double beta1 = 0.9;
	public double beta2 = 0.999;
	protected int epochCount = 0;
	
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
	public String optimizationFunction = "gd";
	public Optimizer optimizer = new Optimizer();
	
	public double regLambda = 0.01;
	public double learningRate = 0.01;
	
	public double learningRateDecayFactor = 1.0;
	public boolean useLRDecay = false;
	public boolean staircase = false;
	
	public int globalStep = 0;
	public int decaySteps = 0;
	
	public double dropoutP;
	
	public boolean useBatchNormalization = false;
	
	public void sgd() {
		// Add regularization terms (b1 and b2 don't have regularization terms)
		dW.plusEquals(this.W.times(regLambda));
		
		// Gradient descent parameter update
		this.W.plusEquals(dW.times(-learningRate));
		this.b.plusEquals(db.times(-learningRate));
	}
	
	public double decayLearningRatePerStep(double lr, double decayFactor, int globalStep, int decaySteps, boolean staircase) {
		double decayedLr = 0.0;
		
		if (staircase) {
			decayedLr = lr * (Math.pow(decayFactor, (int)Math.floor(globalStep/decaySteps)));
		} else {
			decayedLr = lr * (Math.pow(decayFactor, (globalStep/decaySteps)));
		}
		
		return decayedLr;
	}
	
	public double decayLearningRate(double lr, double decayFactor) {
		return lr * decayFactor;
	}
	
	public void adjustLearningRate() {
		if (this.useLRDecay) {
			if (decaySteps > 0) {
				this.learningRate = decayLearningRatePerStep(this.learningRate, this.learningRateDecayFactor, this.globalStep, this.decaySteps, this.staircase);
			} else {
				this.learningRate = decayLearningRate(this.learningRate, this.learningRateDecayFactor);
			}
		}
	}
}
