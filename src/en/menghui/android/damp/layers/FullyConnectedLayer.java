package en.menghui.android.damp.layers;

import java.util.List;

import en.menghui.android.damp.optimizations.AdaDeltaOptimizer;
import en.menghui.android.damp.optimizations.AdaGradOptimizer;
import en.menghui.android.damp.optimizations.AdamOptimizer;
import en.menghui.android.damp.optimizations.GDOptimizer;
import en.menghui.android.damp.optimizations.NetsterovOptimizer;
import en.menghui.android.damp.optimizations.SGDOptimizer;
import en.menghui.android.damp.optimizations.WindowGradOptimizer;
import en.menghui.android.damp.utils.MatrixUtils;
import en.menghui.android.damp.utils.NeuralNetUtils;
import Jama.Matrix;

public class FullyConnectedLayer extends Layer {
	// Batch Normalization parameters.
	Matrix gamma;
	Matrix beta;
	Matrix xmu;
	Matrix var;
	Matrix sqrtvar;
	Matrix ivar;
	Matrix xhat;
	Matrix batchNormOut;
	
	public FullyConnectedLayer(int nIn, int nOut, String actFunc, double dropoutP) {
		this.type = "fully connected";
		this.nIn = nIn;
		this.nOut = nOut;
		this.activationFunction = actFunc;
		this.dropoutP = dropoutP;
		
		// double epsilonInit = 0.12; 
		
		this.W = Matrix.random(nIn, nOut);
		this.W = NeuralNetUtils.initRandomMatrix(this.W);
		this.b = new Matrix(1, nOut, 0.0);
		
		// Memory variables for AdaGrad, Adam, AdaDelta optimizer.
		this.mW = new Matrix(nIn, nOut, 0.0);
		this.mb = new Matrix(1, nOut, 0.0);
		
		// Memory variables for Adam, AdaDelta optimizer.
		this.vW = new Matrix(nIn, nOut, 0.0);
		this.vb = new Matrix(1, nOut, 0.0);
		
		// this.params = NeuralNetUtils.combineMatrixHorizontal(this.W, this.b);
	}
	
	public void forwardProp(Matrix inpt, Matrix dropoutInpt, int miniBatchSize) {
		this.input = inpt;
		
		Matrix h = NeuralNetUtils.add(this.input.times(this.W).times(1.0-this.dropoutP), this.b);
		if (this.useBatchNormalization) {
			double[][] x = {{1.0, 2.0, 5.0}, {3.0, 4.0, 6.0}};
			// Matrix h = new Matrix(x);
			
			h = batchNormalizationForward(h);
		}
		
		
		try {
			// this.output = NeuralNetUtils.sigmoid(NeuralNetUtils.add(this.input.times(this.W).times(1.0-this.dropoutP), this.b), false);
			this.output = NeuralNetUtils.sigmoid(h, false);
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
		Matrix bpIn = bpInput;
		
		
		if (this.useBatchNormalization) {
			// NeuralNetUtils.printMatrix(bpInput);
		    double[][] x = {{1.0, 2.0, 5.0}, {3.0, 4.0, 6.0}};
		    Matrix h = new Matrix(x);
		    bpIn = batchNormalizationBackward(bpInput);
		    // NeuralNetUtils.printMatrix(bnbpInput);
		}
		
		
		Matrix deriv = NeuralNetUtils.sigmoid(this.output, true);
		
		// this.bpOutput = bpIn.times(this.W.transpose()).arrayTimes(deriv);
		this.bpOutput = bpIn.arrayTimes(deriv).times(this.W.transpose());
		
		// Matrix dW = this.input.transpose().times(bpInput);
		// Matrix db = NeuralNetUtils.sum(bpInput, 0);
		// this.dW = this.input.transpose().times(bpIn);
		this.dW = this.input.transpose().times(bpIn.arrayTimes(deriv));
		this.db = NeuralNetUtils.sum(bpIn, 0);
		
		// Add regularization terms (b1 and b2 don't have regularization terms)
		dW.plusEquals(this.W.times(regLambda));
		
		
	}
	
	private Matrix dropout(Matrix inp, double dropoutP) {
		Matrix out = new Matrix(inp.getRowDimension(), inp.getColumnDimension());
		
		return inp;
	}
	
	public void optimize() {
		/* if (this.optimizationFunction.equals("gd")) {
			gd();
		} else if (this.optimizationFunction.equals("adagrad")) {
			adaGrad();
		} else if (this.optimizationFunction.equals("adam")) {
			adam();
		} */
		
		if (this.optimizer instanceof AdamOptimizer) {
			AdamOptimizer optzer = (AdamOptimizer) this.optimizer;
			List<Matrix> pags = optzer.optimize(this.mW, this.vW, this.dW, this.W, this.epochCount);
			this.mW = pags.get(0);
			this.vW = pags.get(1);
			this.dW = pags.get(2);
			this.W  = pags.get(3);
			
			pags = optzer.optimize(this.mb, this.vb, this.db, this.b, this.epochCount);
			this.mb = pags.get(0);
			this.vb = pags.get(1);
			this.db = pags.get(2);
			this.b  = pags.get(3);
			
			this.epochCount++;
		} else if (this.optimizer instanceof AdaGradOptimizer) {
			AdaGradOptimizer optzer = (AdaGradOptimizer) this.optimizer;
			List<Matrix> pags = optzer.optimize(this.mW, this.dW, this.W);
			this.mW = pags.get(0);
			this.dW = pags.get(1);
			this.W  = pags.get(2);
			
			pags = optzer.optimize(this.mb, this.db, this.b);
			this.mb = pags.get(0);
			this.db = pags.get(1);
			this.b  = pags.get(2);
		} else if (this.optimizer instanceof AdaDeltaOptimizer) {
			AdaDeltaOptimizer optzer = (AdaDeltaOptimizer) this.optimizer;
			List<Matrix> pags = optzer.optimize(this.mW, this.vW, this.dW, this.W);
			this.mW = pags.get(0);
			this.vW = pags.get(1);
			this.dW = pags.get(2);
			this.W  = pags.get(3);
			
			pags = optzer.optimize(this.mb, this.vb, this.db, this.b);
			this.mb = pags.get(0);
			this.vb = pags.get(1);
			this.db = pags.get(2);
			this.b  = pags.get(3);
		}  else if (this.optimizer instanceof GDOptimizer) {
			GDOptimizer optzer = (GDOptimizer) this.optimizer;
			List<Matrix> pags = optzer.optimize(this.dW, this.W);
			this.dW = pags.get(0);
			this.W  = pags.get(1);
			
			pags = optzer.optimize(this.db, this.b);
			this.db = pags.get(0);
			this.b  = pags.get(1);
		} else if (this.optimizer instanceof SGDOptimizer) {
			SGDOptimizer optzer = (SGDOptimizer) this.optimizer;
			List<Matrix> pags = optzer.optimize(this.mW, this.dW, this.W);
			this.mW = pags.get(0);
			this.dW = pags.get(1);
			this.W  = pags.get(2);
			
			pags = optzer.optimize(this.mb, this.db, this.b);
			this.mb = pags.get(0);
			this.db = pags.get(1);
			this.b  = pags.get(2);
		} else if (this.optimizer instanceof NetsterovOptimizer) {
			NetsterovOptimizer optzer = (NetsterovOptimizer) this.optimizer;
			List<Matrix> pags = optzer.optimize(this.mW, this.dW, this.W);
			this.mW = pags.get(0);
			this.dW = pags.get(1);
			this.W  = pags.get(2);
			
			pags = optzer.optimize(this.mb, this.db, this.b);
			this.mb = pags.get(0);
			this.db = pags.get(1);
			this.b  = pags.get(2);
		} else if (this.optimizer instanceof WindowGradOptimizer) {
			WindowGradOptimizer optzer = (WindowGradOptimizer) this.optimizer;
			List<Matrix> pags = optzer.optimize(this.mW, this.dW, this.W);
			this.mW = pags.get(0);
			this.dW = pags.get(1);
			this.W  = pags.get(2);
			
			pags = optzer.optimize(this.mb, this.db, this.b);
			this.mb = pags.get(0);
			this.db = pags.get(1);
			this.b  = pags.get(2);
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
	
	public void adaGrad() {
		this.adjustLearningRate();
		
		this.mW.plusEquals(this.dW.arrayTimes(this.dW));
		this.mb.plusEquals(this.db.arrayTimes(this.db));
		
		// Adagrad update.
		Matrix epsilonMat = new Matrix(this.W.getRowDimension(), this.W.getColumnDimension(), 1e-8);
		this.W.plusEquals(this.dW.arrayRightDivide(MatrixUtils.sqrt(this.mW.plus(epsilonMat))).times(this.learningRate).uminus());
		// Wxh.plusEquals(MatrixUtils.sqrt(mWxh.plus(epsilonMat)).arrayLeftDivide(dWxh).times(-learningRate));
		
		epsilonMat = new Matrix(this.b.getRowDimension(), this.b.getColumnDimension(), 1e-8);
		this.b.plusEquals(this.db.arrayRightDivide(MatrixUtils.sqrt(this.mb.plus(epsilonMat))).times(this.learningRate).uminus());
		
		// Reset diffs to zero.
		this.dW = new Matrix(this.W.getRowDimension(), this.W.getColumnDimension(), 0.0);
		this.db = new Matrix(this.b.getRowDimension(), this.b.getColumnDimension(), 0.0);
	}
	
	public void adam() {
		double alpha = 0.01;
		this.learningRate = alpha * Math.sqrt(1.0 - Math.pow(this.beta2, this.epochCount)) / (1.0 - Math.pow(beta1, this.epochCount));
		
		// this.adjustLearningRate();
		
	    this.mW = this.mW.times(this.beta1).plus(this.dW.times(1.0 - this.beta1)); // Update biased first moment estimate.
	    this.mb = this.mb.times(this.beta1).plus(this.db.times(1.0 - this.beta1)); // Update biased first moment estimate.
	    
		this.vW = this.vW.times(this.beta2).plus(this.dW.arrayTimes(this.dW).times(1.0 - this.beta2)); // Update biased second moment estimate.
		this.vb = this.vb.times(this.beta2).plus(this.db.arrayTimes(this.db).times(1.0 - this.beta2)); // Update biased second moment estimate.
		
		Matrix biasCorrW1 = this.mW.times(1.0 - Math.pow(this.beta1, this.epochCount)); // Correct bias first moment estimate.
		Matrix biasCorrb1 = this.mb.times(1.0 - Math.pow(this.beta1, this.epochCount)); // Correct bias first moment estimate.
		Matrix biasCorrW2 = this.vW.times(1.0 - Math.pow(this.beta2, this.epochCount)); // Correct bias second moment estimate.
		Matrix biasCorrb2 = this.vb.times(1.0 - Math.pow(this.beta2, this.epochCount)); // Correct bias second moment estimate.
		
		Matrix epsilonMat = new Matrix(this.W.getRowDimension(), this.W.getColumnDimension(), 1e-8);
		// this.W.plusEquals(this.mW.arrayRightDivide(MatrixUtils.sqrt(this.vW).plus(epsilonMat)).times(-this.learningRate));
		this.W.plusEquals(biasCorrW1.arrayRightDivide(MatrixUtils.sqrt(biasCorrW2).plus(epsilonMat)).times(-this.learningRate));
		
		epsilonMat = new Matrix(this.b.getRowDimension(), this.b.getColumnDimension(), 1e-8);
		// this.b.plusEquals(this.mb.arrayRightDivide(MatrixUtils.sqrt(this.vb).plus(epsilonMat)).times(-this.learningRate));
		this.b.plusEquals(biasCorrb1.arrayRightDivide(MatrixUtils.sqrt(biasCorrb2).plus(epsilonMat)).times(-this.learningRate));
		
		// Reset diffs to zero.
		this.dW = new Matrix(this.W.getRowDimension(), this.W.getColumnDimension(), 0.0);
		this.db = new Matrix(this.b.getRowDimension(), this.b.getColumnDimension(), 0.0);
		
		this.epochCount++;
	}
	
	public Matrix batchNormalizationForward(Matrix h) {
		int N = h.getRowDimension();
		int D = h.getColumnDimension();
		Matrix mu = NeuralNetUtils.sum(h, 0).times(1.0/N);
		xmu = MatrixUtils.minusVector(h, mu);
		Matrix sq = xmu.arrayTimes(xmu); // Square xmu.
		var = NeuralNetUtils.sum(sq, 0).times(1.0/N);
		Matrix eps = new Matrix(var.getRowDimension(), var.getColumnDimension(), 1e-8); 
		sqrtvar = MatrixUtils.sqrt(var.plus(eps));
		Matrix oneMat = new Matrix(sqrtvar.getRowDimension(), sqrtvar.getColumnDimension(), 1.0);
		ivar = oneMat.arrayRightDivide(sqrtvar);
		xhat = MatrixUtils.timesVector(xmu, ivar);
		gamma = new Matrix(xhat.getRowDimension(), xhat.getColumnDimension(), 1.0);
		Matrix gammax = gamma.arrayTimes(xhat);
		beta = new Matrix(gammax.getRowDimension(), gammax.getColumnDimension(), 0.0);
		batchNormOut = gammax.plus(beta);
		
		return batchNormOut;
	}
	
	public Matrix batchNormalizationBackward(Matrix h) {
		int N = h.getRowDimension();
		int D = h.getColumnDimension();
		Matrix dbeta = NeuralNetUtils.sum(h, 0);
		Matrix dgammax = h; // not necessary, but more understandable
		Matrix dgamma = NeuralNetUtils.sum(dgammax.arrayTimes(xhat), 0);
	    Matrix dxhat = dgammax.arrayTimes(gamma);
	    Matrix divar = NeuralNetUtils.sum(dxhat.arrayTimes(xmu), 0);
	    Matrix dxmu1 = MatrixUtils.timesVector(dxhat, ivar);
	    
	    Matrix sqrtMat = sqrtvar.arrayTimes(sqrtvar);
	    Matrix negOneMat = new Matrix(sqrtMat.getRowDimension(), sqrtMat.getColumnDimension(), -1.0);
	    Matrix dsqrtvar = negOneMat.arrayRightDivide(sqrtMat).arrayTimes(divar);
	    
	    Matrix eps = new Matrix(var.getRowDimension(), var.getColumnDimension(), 1e-8);
	    sqrtMat = MatrixUtils.sqrt(var.plus(eps));
	    Matrix oneMat = new Matrix(sqrtMat.getRowDimension(), sqrtMat.getColumnDimension(), 1.0);
	    Matrix dvar = oneMat.arrayRightDivide(sqrtMat).times(0.5).arrayTimes(dsqrtvar);
	    Matrix dsq = MatrixUtils.timesVector(new Matrix(N, D, 1.0).times(1.0/N), dvar);
	    Matrix dxmu2 = xmu.arrayTimes(dsq).times(2);
	    Matrix dx1 = dxmu1.plus(dxmu2);
	    Matrix dmu = NeuralNetUtils.sum(dxmu1.plus(dxmu2), 0).times(-1.0);
	    Matrix dx2 = MatrixUtils.timesVector(new Matrix(N, D, 1.0).times(1.0/N), dmu);
	    Matrix dx = dx1.plus(dx2);
	    
	    return dx;
	}
}
