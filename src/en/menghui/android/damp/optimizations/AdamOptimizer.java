package en.menghui.android.damp.optimizations;

import java.util.ArrayList;
import java.util.List;

import android.util.Log;
import en.menghui.android.damp.utils.MatrixUtils;
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
	
	
}
