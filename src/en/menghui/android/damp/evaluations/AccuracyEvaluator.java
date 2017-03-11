package en.menghui.android.damp.evaluations;

import Jama.Matrix;

public class AccuracyEvaluator extends Evaluator {
	@Override
	public double evaluate(Matrix preds, Matrix targets) {
		int numCorrect = 0;
		
		int n = preds.getRowDimension() * preds.getColumnDimension();
		
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < preds.getColumnDimension(); j++) {
				if ((int)preds.get(i, j) == (int)targets.get(i,  j)) {
					numCorrect++;
				}
			}
		}
		
		if (numCorrect == 0) {
			return 0;
		}
		
		return (double)numCorrect / (double)n;
	}
	
	
}
