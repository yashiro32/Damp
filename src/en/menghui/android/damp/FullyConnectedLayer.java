package en.menghui.android.damp;

import Jama.Matrix;

public class FullyConnectedLayer extends Layer {
	public FullyConnectedLayer(int nIn, int nOut, String actFunc, double dropoutP) {
		this.type = "fully connected";
		this.nIn = nIn;
		this.nOut = nOut;
		this.activationFunction = actFunc;
		this.dropoutP = dropoutP;
		
		// double epsilonInit = 0.12; 
		
		this.W = Matrix.random(nIn, nOut);
		this.b = new Matrix(1, nOut, 0.0);
		
		// this.params = NeuralNetUtils.combineMatrixHorizontal(this.W, this.b);
	}
	
	public void setInput(Matrix inpt, Matrix dropoutInpt, int miniBatchSize) {
		this.input = inpt;
		try {
			this.output = NeuralNetUtils.sigmoid(NeuralNetUtils.add(this.input.times(this.W).times(1.0-this.dropoutP), this.b), false);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		this.yOut = NeuralNetUtils.argmax(this.output, 1);
		
		this.dropoutInput = dropout(dropoutInpt, this.dropoutP);
		try {
			this.dropoutOutput = NeuralNetUtils.sigmoid(NeuralNetUtils.add(this.dropoutInput.times(this.W), this.b), false);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public double accuracy(Matrix y) {
		int correct = 0;
		double accuracy = 0.0;
		
		for (int i = 0; i < y.getRowDimension(); i++) {
			if (y.get(i, 0) == this.yOut.get(i, 0)) {
				correct++;
			}
		}
		
		accuracy = (correct/y.getRowDimension()) * 100.0;
		
		return accuracy;
	}
	
	public void backProp(Matrix bpInput) {
		this.bpInput = bpInput;
		
		Matrix deriv = NeuralNetUtils.sigmoid(this.input, true);
		
		this.bpOutput = bpInput.times(this.W.transpose()).arrayTimes(deriv);
		
		Matrix dW = this.input.transpose().times(bpInput);
		Matrix db = NeuralNetUtils.sum(bpInput, 0);
		
		// Add regularization terms (b1 and b2 don't have regularization terms)
		dW.plusEquals(this.W.times(regLambda));
		
		// Gradient descent parameter update
		this.W.plusEquals(dW.times(-learningRate));
		this.b.plusEquals(db.times(-learningRate));
	}
	
	private Matrix dropout(Matrix inp, double dropoutP) {
		Matrix out = new Matrix(inp.getRowDimension(), inp.getColumnDimension());
		
		return inp;
	}
}
