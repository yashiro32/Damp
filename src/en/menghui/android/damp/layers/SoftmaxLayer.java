package en.menghui.android.damp.layers;

import java.util.List;
import java.util.Random;

import android.util.Log;
import en.menghui.android.damp.optimizations.AdaDeltaOptimizer;
import en.menghui.android.damp.optimizations.AdaGradOptimizer;
import en.menghui.android.damp.optimizations.AdamOptimizer;
import en.menghui.android.damp.optimizations.GDOptimizer;
import en.menghui.android.damp.optimizations.NetsterovOptimizer;
import en.menghui.android.damp.optimizations.SGDOptimizer;
import en.menghui.android.damp.optimizations.WindowGradOptimizer;
import en.menghui.android.damp.utils.MathUtils;
import en.menghui.android.damp.utils.MatrixUtils;
import en.menghui.android.damp.utils.NeuralNetUtils;
import en.menghui.android.damp.utils.RandomUtilities;
import Jama.Matrix;

public class SoftmaxLayer extends Layer {
	private static final String TAG = "Softmax Layer";
	private Matrix targetOneHot;
	
	public SoftmaxLayer(int nIn, int nOut, double dropoutP) {
		this.type = "softmax";
		this.nIn = nIn;
		this.nOut = nOut;
		this.dropoutP = dropoutP;
		
		// double epsilonInit = 0.12; 
		
		this.W = Matrix.random(nIn, nOut);
		this.W = NeuralNetUtils.initRandomMatrix(this.W);
		this.b = new Matrix(1, nOut, 0.0);
		
		// Memory variables for AdaGrad.
		this.mW = new Matrix(nIn, nOut, 0.0);
		this.mb = new Matrix(1, nOut, 0.0);
		
		// Memory variables for Adam, AdaDelta optimizer.
		this.vW = new Matrix(nIn, nOut, 0.0);
		this.vb = new Matrix(1, nOut, 0.0);
		
		// this.params = NeuralNetUtils.combineMatrixHorizontal(this.W, this.b);
	}
	
	public void forwardProp(Matrix inpt, Matrix dropoutInpt, Matrix target, int miniBatchSize) {
		this.input = inpt;
		this.target = target;
		
		this.targetOneHot = new Matrix(target.getRowDimension(), nOut, 0.0);
		for (int i = 0; i < target.getRowDimension(); i++) {
			for (int j = 0; j < nOut; j++) {
				targetOneHot.set(i, j, j == target.get(i, 0) ? 1 : 0);
			}
		}
		
		try {
			this.output = activation.forwardProp(NeuralNetUtils.add(this.input.times(this.W).times(1.0-this.dropoutP), this.b));
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		
		/* this.dropoutInput = dropout(dropoutInpt, this.dropoutP);
		try {
			this.dropoutOutput = activation.forwardProp(NeuralNetUtils.add(this.dropoutInput.times(this.W), this.b));
		} catch (Exception e) {
			e.printStackTrace();
		} */
		
		if (this.useDropout) {
			this.output = this.dropout.forwardProp(this.output, this.isTraining);
			this.dropoutMat = this.dropout.dropoutMat;
		}
		
		
		try {
			this.yOut = NeuralNetUtils.argmax(this.output, 1);
		} catch (Exception e1) {
			e1.printStackTrace();
		}
	}
	
	public double cost() {
		double cost = 0.0;
		
		return cost;
	}
	
	public double accuracy(Matrix y) {
		int correct = 0;
		
		for (int i = 0; i < y.getRowDimension(); i++) {
			if ((int)this.target.get(i, 0) == (int)this.yOut.get(i, 0)) {
				correct++;
			}
		}
		
		double accuracy = ((double)correct/(double)y.getRowDimension()) * 100.0;
		
		return accuracy;
	}
	
	public void backProp(Matrix bpInput) {
		this.bpInput = bpInput;
		
		Matrix error = this.output.minus(this.targetOneHot);
		
		/* Matrix delta = this.output.copy();
		for (int k = 0; k < delta.getRowDimension(); k++) {
			delta.set(k, (int)target.get(k, 0), delta.get(k, (int)target.get(k, 0)) - 1.0);
		} */
		
		Matrix deriv = activation.backProp(this.output);
		
		
		Matrix delta;
		if (useDropout) {
			delta = error.arrayTimes(deriv).arrayTimes(this.dropoutMat);
		} else {
			delta = error.arrayTimes(deriv);
		}
		
		
		// this.bpOutput = delta.times(this.W.transpose()).arrayTimes(deriv);
	    this.bpOutput = delta.times(this.W.transpose());
		
		// Matrix dW = this.input.transpose().times(delta);
		// Matrix db = NeuralNetUtils.sum(delta, 0);
		// this.dW = this.input.transpose().times(delta);
		this.dW = this.input.transpose().times(delta);
		this.db = NeuralNetUtils.sum(error, 0);
		
		// Add regularization terms (b1 and b2 don't have regularization terms)
		dW.plusEquals(this.W.times(regLambda));
		
		
	}
	
	
	public void optimize() {
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
		} else if (this.optimizer instanceof GDOptimizer) {
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
	
	
}
