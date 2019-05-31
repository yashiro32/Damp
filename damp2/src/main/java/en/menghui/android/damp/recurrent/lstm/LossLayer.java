package en.menghui.android.damp.recurrent.lstm;

import Jama.Matrix;

public class LossLayer {
	public static Matrix bottomDiff(Matrix pred, double label) {
		Matrix diff = new Matrix (pred.getRowDimension(), pred.getColumnDimension(), 0.0);
		
		diff.set(0, 0, 2.0 * (pred.get(0, 0) - label));
		
		return diff;
	}
}
