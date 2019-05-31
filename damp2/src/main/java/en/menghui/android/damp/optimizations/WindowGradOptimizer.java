package en.menghui.android.damp.optimizations;

import java.util.ArrayList;
import java.util.List;

import Jama.Matrix;
import android.util.Log;
import en.menghui.android.damp.utils.MatrixUtils;

public class WindowGradOptimizer extends Optimizer {
	private static final String TAG = "WindowGrad Optimizer";
	public double ro = 0.95;
	public double eps = 1e-6;
	
	public WindowGradOptimizer(double ro, double eps, double lr) {
		this.ro = ro;
		this.eps = eps;
		this.learningRate = lr;
	}
	
	public List<Matrix> optimize(Matrix m, Matrix d, Matrix p) {
		List<Matrix> list = new ArrayList<Matrix>();
		
		// this is adagrad but with a moving window weighted average
        // so the gradient is not accumulated over the entire history of the run. 
        // it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
        m = m.times(this.ro).plus(d.arrayTimes(d).times(1 - this.ro));
		Matrix epsilonMat = new Matrix(p.getRowDimension(), p.getColumnDimension(), this.eps);
		Matrix lrMat = new Matrix(p.getRowDimension(), p.getColumnDimension(), -this.learningRate);
		Matrix dx = lrMat.arrayRightDivide(MatrixUtils.sqrt(m.plus(epsilonMat))).arrayTimes(d); // eps added for better conditioning
		p.plusEquals(dx);
		
		// Reset diffs to zero.
		d = new Matrix(p.getRowDimension(), p.getColumnDimension(), 0.0);
		
		list.add(m);
		list.add(d);
		list.add(p);
		
		Log.d(TAG, "This layer is using WindowGrad optimization technique.");
		
		return list;
	}
	
	
}
