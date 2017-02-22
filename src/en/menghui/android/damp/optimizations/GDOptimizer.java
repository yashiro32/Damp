package en.menghui.android.damp.optimizations;

import java.util.ArrayList;
import java.util.List;

import Jama.Matrix;
import android.util.Log;

public class GDOptimizer extends Optimizer {
	private static final String TAG = "GD Optimizer";
	
	public GDOptimizer(double lr) {
		this.learningRate = lr;
	}
	
	public List<Matrix> optimize(Matrix d, Matrix p) {
		List<Matrix> list = new ArrayList<Matrix>();
		
		this.adjustLearningRate();
		
		// Gradient descent parameter update
		p.plusEquals(d.times(-this.learningRate));
		
		list.add(d);
		list.add(p);
		
		Log.d(TAG, "This layer is using Gradient Descent optimization technique.");
		
		return list;
	}
}
