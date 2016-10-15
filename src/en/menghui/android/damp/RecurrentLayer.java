package en.menghui.android.damp;

import Jama.Matrix;

public class RecurrentLayer {
	public String type = "";
	
	public int nIn;
	public int nOut;
	public int nHidden = 0;
	
	public Matrix Wxh;
	public Matrix Whh;
	public Matrix Why;
	public Matrix bh;
	public Matrix by;
	
	public Matrix dWxh;
	public Matrix dWhh;
	public Matrix dWhy;
	public Matrix dbh;
	public Matrix dby;
	
	public Matrix mWxh;
	public Matrix mWhh;
	public Matrix mWhy;
	public Matrix mbh;
	public Matrix mby;
	
	public String activationFunction;
	
	public double regLambda = 0.01;
	public double learningRate = 1e-1;
	
	public double dropoutP;
	
	public Matrix[] inputs;
	public Matrix[] targets;
	public Matrix hprev;
	
	private Matrix[] xs;
	private Matrix[] ps;
	private Matrix[] hs;
	
	public double loss = 0.0;
	
	public RecurrentLayer(int nIn, int nOut, String actFunc, double dropoutP, int nHidden) {
		this.type = "recurrent";
		
		this.nIn = nIn;
		this.nOut = nOut;
		this.nHidden = nHidden;
		this.activationFunction = actFunc;
		this.dropoutP = dropoutP;
		
		// double epsilonInit = 0.12; 
		
		this.Wxh = Matrix.random(nHidden, nIn); // Input to Hidden
		this.Whh = Matrix.random(nHidden, nHidden); // Hidden to Hidden
		this.Why = Matrix.random(nOut, nHidden); // Hidden to Output
		this.bh = new Matrix(nHidden, 1, 0.0); // Hidden Bias
		this.by = new Matrix(nOut, 1, 0.0); // Output Bias
		
		this.Wxh.timesEquals(0.01);
		this.Whh.timesEquals(0.01);
		this.Why.timesEquals(0.01);
		
		this.mWxh = new Matrix(this.Wxh.getRowDimension(), this.Wxh.getColumnDimension(), 0.0);
		this.mWhh = new Matrix(this.Whh.getRowDimension(), this.Whh.getColumnDimension(), 0.0);
		this.mWhy = new Matrix(this.Why.getRowDimension(), this.Why.getColumnDimension(), 0.0);
		this.mbh = new Matrix(this.bh.getRowDimension(), this.bh.getColumnDimension(), 0.0);
		this.mby = new Matrix(this.by.getRowDimension(), this.by.getColumnDimension(), 0.0);
	}
	
	public void setArgumentsForPropagation(Matrix[] inputs, Matrix[] targets, Matrix hprev) {
		this.inputs = inputs;
		this.targets = targets;
		this.hprev = hprev;
	}
	
	public void forwardProp() {
		xs = new Matrix[inputs.length];
		hs = new Matrix[inputs.length];
		Matrix[] ys = new Matrix[inputs.length];
		ps = new Matrix[inputs.length];
		
		// hs[0] = hprev.copy();
		
		for (int t = 0; t < inputs.length; t++) {
			xs[t] = new Matrix(nIn, 1, 0.0);
			
			for (int k = 0; k < inputs[t].getRowDimension(); k++) {
				int index = (int)inputs[t].get(k, 0);
				xs[t].set(index, 0, 1.0);
			}
			
			if (t == 0) {
				hs[t] = NeuralNetUtils.tanh(Wxh.times(xs[t]).plus(Whh.times(hprev).plus(bh)), false);
			} else {
				hs[t] = NeuralNetUtils.tanh(Wxh.times(xs[t]).plus(Whh.times(hs[t-1]).plus(bh)), false);
			}
			
			ys[t] = Why.times(hs[t]).plus(by); // unnormalized log probabilities for next chars
			
			double expSum = NeuralNetUtils.sum(NeuralNetUtils.exp(ys[t]), 0).get(0, 0);
			ps[t] = NeuralNetUtils.scalarLeftDivide(expSum, NeuralNetUtils.exp(ys[t]));
			
			// loss += NeuralNetUtils.log(ps[targets, 0]); // softmax (cross-entropy loss)
		}
	}
	
	public void backProp() {
		this.dWxh = new Matrix(this.Wxh.getRowDimension(), this.Wxh.getColumnDimension(), 0.0);
		this.dWhh = new Matrix(this.Whh.getRowDimension(), this.Whh.getColumnDimension(), 0.0);
		this.dWhy = new Matrix(this.Why.getRowDimension(), this.Why.getColumnDimension(), 0.0);
		this.dbh = new Matrix(this.bh.getRowDimension(), this.bh.getColumnDimension(), 0.0);
		this.dby = new Matrix(this.by.getRowDimension(), this.by.getColumnDimension(), 0.0);
		
		Matrix dhnext = new Matrix(hs[0].getRowDimension(), hs[0].getColumnDimension(), 0.0);
		Matrix oneMat = new Matrix(hs[0].getRowDimension(), hs[0].getColumnDimension(), 1.0);
		
		for (int t = inputs.length-1; t > -1; t--) {
			Matrix dy = ps[t].copy();
			
			// Backprop into y
			for (int i = 0; i < targets[t].getRowDimension(); i++) {
				for (int j = 0; j < targets[t].getColumnDimension(); j++) {
					int index = (int)targets[t].get(i, j);
					dy.set(index, 0, dy.get(index, 0) - 1.0);
				}
			}
			
			dWhy.plusEquals(dy.times(hs[t].transpose()));
			dby.plusEquals(dy);
			
			Matrix dh = Why.transpose().times(dy).plus(dhnext); // Backprop into h
			Matrix dhraw = oneMat.minus(hs[t].arrayTimes(hs[t])).arrayTimes(dh); // Backprop through tanh nonlinearity
			dbh.plusEquals(dhraw);
			dWxh.plusEquals(dhraw.times(xs[t].transpose()));
			
			if (t == 0) {
				dWhh.plusEquals(dhraw.times(hprev.transpose()));
			} else {
				dWhh.plusEquals(dhraw.times(hs[t-1].transpose()));
			}
			
			dhnext = Whh.transpose().times(dhraw);
			
		}
		
		// Set hprev to the latest hs
		hprev = hs[hs.length-1];
	}
	
	public void adaGrad() {
		mWxh.plusEquals(dWxh.arrayTimes(dWxh));
		mWhh.plusEquals(dWhh.arrayTimes(dWhh));
		mWhy.plusEquals(dWhy.arrayTimes(dWhy));
		mbh.plusEquals(dbh.arrayTimes(dbh));
		mby.plusEquals(dby.arrayTimes(dby));
		
		Matrix epsilonMat = new Matrix(Wxh.getRowDimension(), Wxh.getColumnDimension(), 1e-8);
		Wxh.plusEquals(dWxh.arrayRightDivide(MatrixUtils.sqrt(mWxh.plus(epsilonMat))).times(-learningRate));
		// Wxh.plusEquals(MatrixUtils.sqrt(mWxh.plus(epsilonMat)).arrayLeftDivide(dWxh).times(-learningRate));
		
		Matrix a = new Matrix(new double[] {1, 2, 3, 4}, 4);
		Matrix b = new Matrix(new double[] {5, 6, 9, 16}, 4);
		NeuralNetUtils.printMatrix(dWxh.arrayRightDivide(MatrixUtils.sqrt(mWxh.plus(epsilonMat))));
		
		epsilonMat = new Matrix(Whh.getRowDimension(), Whh.getColumnDimension(), 1e-8);
		Whh.plusEquals(dWhh.arrayLeftDivide(MatrixUtils.sqrt(mWhh.plus(epsilonMat))).times(-learningRate));
		
		epsilonMat = new Matrix(Why.getRowDimension(), Why.getColumnDimension(), 1e-8);
		Why.plusEquals(dWhy.arrayLeftDivide(MatrixUtils.sqrt(mWhy.plus(epsilonMat))).times(-learningRate));
		
		epsilonMat = new Matrix(bh.getRowDimension(), bh.getColumnDimension(), 1e-8);
		bh.plusEquals(dbh.arrayLeftDivide(MatrixUtils.sqrt(mbh.plus(epsilonMat))).times(-learningRate));
		
		epsilonMat = new Matrix(by.getRowDimension(), by.getColumnDimension(), 1e-8);
		by.plusEquals(dby.arrayLeftDivide(MatrixUtils.sqrt(mby.plus(epsilonMat))).times(-learningRate));
		
		// NeuralNetUtils.printMatrix(Wxh);
	}
}
