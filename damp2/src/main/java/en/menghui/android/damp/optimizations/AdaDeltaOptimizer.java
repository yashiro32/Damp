package en.menghui.android.damp.optimizations;

import java.util.ArrayList;
import java.util.List;

import android.util.Log;
import en.menghui.android.damp.utils.MatrixUtils;
import Jama.Matrix;

public class AdaDeltaOptimizer extends Optimizer {
	private static final String TAG = "Ada Delta Optimizer";
	public double ro = 0.95;
	public double eps = 1e-6;
	
	public AdaDeltaOptimizer(double ro, double eps, double lr) {
		this.ro = ro;
		this.eps = eps;
		this.learningRate = lr;
	}
	
	public List<Matrix> optimize(Matrix m, Matrix v, Matrix d, Matrix p) {
		List<Matrix> list = new ArrayList<>();
		
		Matrix epsilonMat = new Matrix(p.getRowDimension(), p.getColumnDimension(), this.eps);
		m = m.times(this.ro).plus(d.arrayTimes(d).times(1 - this.ro));
		Matrix dx = MatrixUtils.sqrt(v.plus(epsilonMat).arrayRightDivide(m.plus(epsilonMat))).uminus().arrayTimes(d);
		v = v.times(this.ro).plus(dx.arrayTimes(dx).times(1 - this.ro));
		p.plusEquals(dx);
		
		// Reset diffs to zero.
		d = new Matrix(p.getRowDimension(), p.getColumnDimension(), 0.0);
		
		list.add(m);
		list.add(v);
		list.add(d);
		list.add(p);
		
		Log.i(TAG, "This layer is using Ada Delta optimization technique.");
		
		return list;
	}
	
	
}
