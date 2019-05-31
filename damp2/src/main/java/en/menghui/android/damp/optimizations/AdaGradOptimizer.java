package en.menghui.android.damp.optimizations;

import java.util.ArrayList;
import java.util.List;

import android.util.Log;
import en.menghui.android.damp.utils.MatrixUtils;
import Jama.Matrix;

public class AdaGradOptimizer extends Optimizer {
	private static final String TAG = "AdaGrad Optimizer";
	
	public AdaGradOptimizer(double lr) {
		this.learningRate = lr;
	}
	
 	public List<Matrix> optimize(Matrix m, Matrix d, Matrix p) {
		List<Matrix> list = new ArrayList<>();
		
		this.adjustLearningRate();
		
		m.plusEquals(d.arrayTimes(d));
		
		// Adagrad update.
		Matrix epsilonMat = new Matrix(p.getRowDimension(), p.getColumnDimension(), 1e-8);
		Matrix lrMat = new Matrix(p.getRowDimension(), p.getColumnDimension(), -this.learningRate);
		// p.plusEquals(d.arrayRightDivide(MatrixUtils.sqrt(m.plus(epsilonMat))).times(this.learningRate).uminus());
		p.plusEquals(lrMat.arrayRightDivide(MatrixUtils.sqrt(m.plus(epsilonMat))).arrayTimes(d));
		// Wxh.plusEquals(MatrixUtils.sqrt(mWxh.plus(epsilonMat)).arrayLeftDivide(dWxh).times(-learningRate));
		
		// Reset diffs to zero.
		d = new Matrix(p.getRowDimension(), p.getColumnDimension(), 0.0);
		
		list.add(m);
		list.add(d);
		list.add(p);
		
		Log.i(TAG, "This layer is using AdaGrad optimization technique.");
		
		return list;
	}
}
