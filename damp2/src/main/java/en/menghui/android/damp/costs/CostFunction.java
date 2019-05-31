package en.menghui.android.damp.costs;

import Jama.Matrix;

public abstract class CostFunction {
	public abstract double calLoss(Matrix preds, Matrix targets);
}
