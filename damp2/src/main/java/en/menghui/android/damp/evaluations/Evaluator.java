package en.menghui.android.damp.evaluations;

import Jama.Matrix;

public abstract class Evaluator {
	public abstract double evaluate(Matrix preds, Matrix targets);
}
