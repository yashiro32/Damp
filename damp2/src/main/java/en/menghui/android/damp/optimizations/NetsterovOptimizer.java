package en.menghui.android.damp.optimizations;

import java.util.ArrayList;
import java.util.List;

import Jama.Matrix;
import android.util.Log;

public class NetsterovOptimizer extends Optimizer {
	private static final String TAG = "Netsterov Optimizer";
	
	public double momentum = 0.9;
	
	public NetsterovOptimizer(double momentum, double lr) {
		this.momentum = momentum;
		this.learningRate = lr;
	}
	
	public List<Matrix> optimize(Matrix m, Matrix d, Matrix p) {
		List<Matrix> list = new ArrayList<Matrix>();
		
        Matrix dx = m;
		m = m.times(this.momentum).plus(d.times(this.learningRate));
		dx = dx.times(this.momentum).minus(m.times(1.0 + this.momentum));
		p.plusEquals(dx);
		 
		// Reset diffs to zero.
		d = new Matrix(p.getRowDimension(), p.getColumnDimension(), 0.0);
		
		list.add(m);
		list.add(d);
		list.add(p);
		
		Log.d(TAG, "This layer is using Netsterov optimization technique.");
		
		return list;
	}
	
	
}
