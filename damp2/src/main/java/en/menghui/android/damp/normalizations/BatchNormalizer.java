package en.menghui.android.damp.normalizations;

import Jama.Matrix;
import en.menghui.android.damp.utils.MatrixUtils;
import en.menghui.android.damp.utils.NeuralNetUtils;

public class BatchNormalizer extends Normalizer {
	// Batch Normalization parameters.
	public double gamma;
	public double beta;
	public Matrix gammaMat;
	public Matrix betaMat;
	public Matrix xmu;
	public Matrix var;
	public Matrix sqrtvar;
	public Matrix ivar;
	public Matrix xhat;
	public Matrix batchNormOut;
	
	public BatchNormalizer(double beta, double gamma) {
		this.beta = beta;
		this.gamma = gamma;
	}
	
	public Matrix forwardProp(Matrix h) {
		int N = h.getRowDimension();
		int D = h.getColumnDimension();
		Matrix mu = NeuralNetUtils.sum(h, 0).times(1.0/N);
		xmu = MatrixUtils.minusVector(h, mu);
		Matrix sq = xmu.arrayTimes(xmu); // Square xmu.
		var = NeuralNetUtils.sum(sq, 0).times(1.0/N);
		Matrix eps = new Matrix(var.getRowDimension(), var.getColumnDimension(), 1e-8); 
		sqrtvar = MatrixUtils.sqrt(var.plus(eps));
		Matrix oneMat = new Matrix(sqrtvar.getRowDimension(), sqrtvar.getColumnDimension(), 1.0);
		ivar = oneMat.arrayRightDivide(sqrtvar);
		xhat = MatrixUtils.timesVector(xmu, ivar);
		gammaMat = new Matrix(xhat.getRowDimension(), xhat.getColumnDimension(), 1.0);
		Matrix gammax = gammaMat.arrayTimes(xhat);
		betaMat = new Matrix(gammax.getRowDimension(), gammax.getColumnDimension(), 0.0);
		batchNormOut = gammax.plus(betaMat);
		
		return batchNormOut;
	}
	
	public Matrix backProp(Matrix h) {
		int N = h.getRowDimension();
		int D = h.getColumnDimension();
		Matrix dbeta = NeuralNetUtils.sum(h, 0);
		Matrix dgammax = h; // not necessary, but more understandable
		Matrix dgamma = NeuralNetUtils.sum(dgammax.arrayTimes(xhat), 0);
	    Matrix dxhat = dgammax.arrayTimes(gammaMat);
	    Matrix divar = NeuralNetUtils.sum(dxhat.arrayTimes(xmu), 0);
	    Matrix dxmu1 = MatrixUtils.timesVector(dxhat, ivar);
	    
	    Matrix sqrtMat = sqrtvar.arrayTimes(sqrtvar);
	    Matrix negOneMat = new Matrix(sqrtMat.getRowDimension(), sqrtMat.getColumnDimension(), -1.0);
	    Matrix dsqrtvar = negOneMat.arrayRightDivide(sqrtMat).arrayTimes(divar);
	    
	    Matrix eps = new Matrix(var.getRowDimension(), var.getColumnDimension(), 1e-8);
	    sqrtMat = MatrixUtils.sqrt(var.plus(eps));
	    Matrix oneMat = new Matrix(sqrtMat.getRowDimension(), sqrtMat.getColumnDimension(), 1.0);
	    Matrix dvar = oneMat.arrayRightDivide(sqrtMat).times(0.5).arrayTimes(dsqrtvar);
	    Matrix dsq = MatrixUtils.timesVector(new Matrix(N, D, 1.0).times(1.0/N), dvar);
	    Matrix dxmu2 = xmu.arrayTimes(dsq).times(2);
	    Matrix dx1 = dxmu1.plus(dxmu2);
	    Matrix dmu = NeuralNetUtils.sum(dxmu1.plus(dxmu2), 0).times(-1.0);
	    Matrix dx2 = MatrixUtils.timesVector(new Matrix(N, D, 1.0).times(1.0/N), dmu);
	    Matrix dx = dx1.plus(dx2);
	    
	    return dx;
	}
}
