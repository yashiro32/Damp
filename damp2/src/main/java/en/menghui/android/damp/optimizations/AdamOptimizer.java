package en.menghui.android.damp.optimizations;

import java.util.ArrayList;
import java.util.List;

import android.util.Log;
import en.menghui.android.damp.utils.MatrixUtils;
import en.menghui.android.damp.utils.Volume;
import Jama.Matrix;

public class AdamOptimizer extends Optimizer {
	private static final String TAG = "Adam Optimizer";
	public double beta1 = 0.9;
	public double beta2 = 0.999;
	protected int epochCount = 0;
	
	public AdamOptimizer(double beta1, double beta2, double lr) {
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.learningRate = lr;
	}
	
	public List<Matrix> optimize(Matrix m, Matrix v, Matrix d, Matrix p, int epochCount) {
		List<Matrix> list = new ArrayList<Matrix>();
		
		double alpha = 0.01;
		this.learningRate = alpha * Math.sqrt(1.0 - Math.pow(this.beta2, epochCount)) / (1.0 - Math.pow(beta1, epochCount));
		
		// this.adjustLearningRate();
		
	    m = m.times(this.beta1).plus(d.times(1.0 - this.beta1)); // Update biased first moment estimate.
	    
		v = v.times(this.beta2).plus(d.arrayTimes(d).times(1.0 - this.beta2)); // Update biased second moment estimate.
		
		Matrix biasCorr1 = m.times(1.0 - Math.pow(this.beta1, epochCount)); // Correct bias first moment estimate.
		Matrix biasCorr2 = v.times(1.0 - Math.pow(this.beta2, epochCount)); // Correct bias second moment estimate.
		
		Matrix epsilonMat = new Matrix(p.getRowDimension(), p.getColumnDimension(), 1e-8);
		// this.W.plusEquals(this.mW.arrayRightDivide(MatrixUtils.sqrt(this.vW).plus(epsilonMat)).times(-this.learningRate));
		p.plusEquals(biasCorr1.arrayRightDivide(MatrixUtils.sqrt(biasCorr2).plus(epsilonMat)).times(-this.learningRate));
		
		// Reset diffs to zero.
		d = new Matrix(p.getRowDimension(), p.getColumnDimension(), 0.0);
		
	    list.add(m);
		list.add(v);
		list.add(d);
		list.add(p);
		
		Log.d(TAG, "This layer is using Adam optimization technique.");
		
		return list;
	}
	
	public List<List<Volume>> optimize(List<Volume> m, List<Volume> v, List<Volume> pag, int epochCount) {
		List<List<Volume>> list = new ArrayList<List<Volume>>();
		
		for (int i = 0; i < m.size(); i++) {
			double[] mW = m.get(i).weights;
			double[] vW = v.get(i).weights;
			double[] dW = pag.get(i).weightGradients;
			double[] W = pag.get(i).weights;
			
			int plen = W.length;
			for (int j = 0; j < plen; j++) {
				// this.l2DecayLoss += l2Decay * parameters[j] * parameters[j] / 2; // Accumulate weight decay loss.
				// this.l1DecayLoss += l1Decay * Math.abs(parameters[j]);
				// double l1Grad = l1Decay * (parameters[j] > 0 ? 1 : -1);
				// double l2Grad = l2Decay * parameters[j];
				
				// double gij = (l2Grad + l1Grad + gradients[j]) / this.batchSize; // Raw batch gradient.
				double gij = dW[j]; // Raw batch gradient.
				
				mW[j] = mW[j] * this.beta1 + (1 - this.beta1) * gij; // Update biased first moment estimate.
				vW[j] = vW[j] * this.beta2 + (1 - this.beta2) * gij * gij; // Update biased second moment estimate.
				double biasCorr1 = mW[j] * (1 - Math.pow(this.beta1, this.epochCount)); // Correct bias first moment estimate.
				double biasCorr2 = vW[j] * (1 - Math.pow(this.beta2, this.epochCount)); // Correct bias second moment estimate.
				double dx = -this.learningRate * biasCorr1 / (Math.sqrt(biasCorr2) + 1e-8);
				W[j] += dx;
				
				dW[j] = 0.0; // Zero out gradient so that we can begin accumulate anew. 
			}
			
		}
		
	    list.add(m);
		list.add(v);
		list.add(pag);
		
		Log.d(TAG, "This layer is using Adam optimization technique.");
		
		return list;
	}
	
	public List<Volume> optimize(Volume m, Volume v, Volume pag, int epochCount) {
		List<Volume> list = new ArrayList<Volume>();
		
		int plen = pag.weights.length;
		for (int j = 0; j < plen; j++) {
			// this.l2DecayLoss += l2Decay * parameters[j] * parameters[j] / 2; // Accumulate weight decay loss.
			// this.l1DecayLoss += l1Decay * Math.abs(parameters[j]);
			// double l1Grad = l1Decay * (parameters[j] > 0 ? 1 : -1);
			// double l2Grad = l2Decay * parameters[j];
			
			// double gij = (l2Grad + l1Grad + gradients[j]) / this.batchSize; // Raw batch gradient.
			double gij = pag.weightGradients[j]; // Raw batch gradient.
			
			m.weights[j] = m.weights[j] * this.beta1 + (1 - this.beta1) * gij; // Update biased first moment estimate.
			v.weights[j] = v.weights[j] * this.beta2 + (1 - this.beta2) * gij * gij; // Update biased second moment estimate.
			double biasCorr1 = m.weights[j] * (1 - Math.pow(this.beta1, this.epochCount)); // Correct bias first moment estimate.
			double biasCorr2 = v.weights[j] * (1 - Math.pow(this.beta2, this.epochCount)); // Correct bias second moment estimate.
			double dx = -this.learningRate * biasCorr1 / (Math.sqrt(biasCorr2) + 1e-8);
			pag.weights[j] += dx;
			
			pag.weightGradients[j] = 0.0; // Zero out gradient so that we can begin accumulate anew. 
		}
		
	    list.add(m);
		list.add(v);
		list.add(pag);
		
		Log.d(TAG, "This layer is using Adam optimization technique.");
		
		return list;
	}
	
	
}
