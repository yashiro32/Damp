package en.menghui.android.damp.costs;

import en.menghui.android.damp.utils.MatrixUtils;
import Jama.Matrix;

public class LRCostFunction extends CostFunction {
	@Override
	public double calLoss(Matrix preds, Matrix targets) {
		double loss = 0;
		
		Matrix oneMat = new Matrix(preds.getRowDimension(), preds.getColumnDimension());
		Matrix log = targets.uminus().transpose().times(MatrixUtils.log(preds)).minus(oneMat.minus(targets).transpose().times(MatrixUtils.log(oneMat.minus(preds))));
		loss = (1 / preds.getRowDimension()) * log.get(0, 0);
		
		return loss;
	}
	
	
}
