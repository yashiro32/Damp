package en.menghui.android.damp.layers;

import en.menghui.android.damp.utils.NeuralNetUtils;
import Jama.Matrix;

public class SoftmaxLayer extends Layer {
	public SoftmaxLayer(int nIn, int nOut, double dropoutP) {
		this.type = "softmax";
		this.nIn = nIn;
		this.nOut = nOut;
		this.dropoutP = dropoutP;
		
		// double epsilonInit = 0.12; 
		
		this.W = Matrix.random(nIn, nOut);
		this.W = NeuralNetUtils.initRandomMatrix(this.W);
		this.b = new Matrix(1, nOut, 0.0);
		
		// this.params = NeuralNetUtils.combineMatrixHorizontal(this.W, this.b);
	}
	
	public void setInput(Matrix inpt, Matrix dropoutInpt, Matrix target, int miniBatchSize) {
		this.input = inpt;
		this.target = target;
		
		try {
			this.output = NeuralNetUtils.sigmoid(NeuralNetUtils.add(this.input.times(this.W).times(1.0-this.dropoutP), this.b), false);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		try {
			this.yOut = NeuralNetUtils.argmax(this.output, 1);
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		
		this.dropoutInput = dropout(dropoutInpt, this.dropoutP);
		
		try {
			this.dropoutOutput = NeuralNetUtils.sigmoid(NeuralNetUtils.add(this.dropoutInput.times(this.W), this.b), false);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public double cost() {
		double cost = 0.0;
		
		return cost;
	}
	
	public double accuracy(Matrix y) {
		int correct = 0;
		
		for (int i = 0; i < y.getRowDimension(); i++) {
			if (y.get(i, 0) == this.yOut.get(i, 0)) {
				correct++;
			}
		}
		
		double accuracy = (correct/y.getRowDimension()) * 100.0;
		
		return accuracy;
	}
	
	public void backProp(Matrix bpInput) {
		this.bpInput = bpInput;
		
		// Matrix delta = this.output.minus(this.target);
		
		Matrix delta = this.output.copy();
		for (int k = 0; k < delta.getRowDimension(); k++) {
			delta.set(k, (int)target.get(k, 0), delta.get(k, (int)target.get(k, 0)) - 1.0);
		}
		
		Matrix deriv = NeuralNetUtils.sigmoid(this.input, true);
		
		this.bpOutput = delta.times(this.W.transpose()).arrayTimes(deriv);
		
		// Matrix dW = this.input.transpose().times(delta);
		// Matrix db = NeuralNetUtils.sum(delta, 0);
		this.dW = this.input.transpose().times(delta);
		this.db = NeuralNetUtils.sum(delta, 0);
		
		// Add regularization terms (b1 and b2 don't have regularization terms)
		dW.plusEquals(this.W.times(regLambda));
		
		
	}
	
	private Matrix dropout(Matrix inp, double dropoutP) {
		Matrix out = new Matrix(inp.getRowDimension(), inp.getColumnDimension());
		
		return inp;
	}
	
	public void optimize() {
		if (this.optimizationFunction.equals("gd")) {
			gd();
		}
	}
	
	public void gd() {
		if (this.useLRDecay) {
			if (decaySteps > 0) {
				this.learningRate = decayLearningRatePerStep(this.learningRate, this.learningRateDecayFactor, this.globalStep, this.decaySteps, this.staircase);
			} else {
				this.learningRate = decayLearningRate(this.learningRate, this.learningRateDecayFactor);
			}
		}
		
		// Gradient descent parameter update
		this.W.plusEquals(this.dW.times(-this.learningRate));
		this.b.plusEquals(this.db.times(-this.learningRate));
	}
}
