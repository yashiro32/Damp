package en.menghui.android.damp.optimizations;

import java.util.ArrayList;
import java.util.List;

import android.util.Log;
import Jama.Matrix;

public class SGDOptimizer extends Optimizer {
	private static final String TAG = "SGD Optimizer";
	public double momentum = 0.9;
	
	public SGDOptimizer(double momentum, double lr) {
		this.momentum = momentum;
		this.learningRate = lr;
	}
	
	public List<Matrix> optimize(Matrix m, Matrix d, Matrix p) {
		List<Matrix> list = new ArrayList<Matrix>();
		
		if (this.momentum > 0.0) {
			// Momentum update.
			Matrix dx = m.times(this.momentum).minus(d.times(this.learningRate)); // Step.
			m = dx; // Back this up for next iteration of momentum.
			p.plusEquals(dx); // Apply corrected gradient.
		} else {
			// Vanilla sgd.
			p.plusEquals(d.times(-this.learningRate));
		}
		
		// Reset diffs to zero.
		d = new Matrix(p.getRowDimension(), p.getColumnDimension(), 0.0);
		
		list.add(m);
		list.add(d);
		list.add(p);
		
		Log.i(TAG, "This layer is using SGD optimization technique.");
		
		return list;
	}
}
