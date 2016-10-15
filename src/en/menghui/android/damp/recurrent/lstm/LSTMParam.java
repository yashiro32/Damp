package en.menghui.android.damp.recurrent.lstm;

import Jama.Matrix;
import en.menghui.android.damp.MatrixUtils;
import en.menghui.android.damp.NeuralNetUtils;

public class LSTMParam {
	public int xDim;
	public int memCellCt;
	
	public Matrix Wg;
	public Matrix Wi;
	public Matrix Wf;
	public Matrix Wo;
	public Matrix bg;
	public Matrix bi;
	public Matrix bf;
	public Matrix bo;
	
	public Matrix dWg;
	public Matrix dWi;
	public Matrix dWf;
	public Matrix dWo;
	public Matrix dbg;
	public Matrix dbi;
	public Matrix dbf;
	public Matrix dbo;
	
	public Matrix mWg;
	public Matrix mWi;
	public Matrix mWf;
	public Matrix mWo;
	public Matrix mbg;
	public Matrix mbi;
	public Matrix mbf;
	public Matrix mbo;
	
	public double regLambda = 0.01;
	public double learningRate = 0.1;
	
	public double learningRateDecayFactor = 1.0;
	public boolean useLRDecay = false;
	public boolean staircase = false;
	
	public int globalStep = 0;
	public int decaySteps = 0;
	
	public LSTMParam(int memCellCt, int xDim) {
		this.memCellCt = memCellCt;
		this.xDim = xDim;
		int concatLen = xDim + memCellCt;
		
		// Weight Matrixes.
		/* this.Wg = Matrix.random(memCellCt, concatLen);
		this.Wi = Matrix.random(memCellCt, concatLen);
	    this.Wf = Matrix.random(memCellCt, concatLen);
		this.Wo = Matrix.random(memCellCt, concatLen); */
		
		this.Wg = NeuralNetUtils.initRandomMatrix(memCellCt, concatLen, -0.1, 0.1);
		this.Wi = NeuralNetUtils.initRandomMatrix(memCellCt, concatLen, -0.1, 0.1);
		this.Wf = NeuralNetUtils.initRandomMatrix(memCellCt, concatLen, -0.1, 0.1);
		this.Wo = NeuralNetUtils.initRandomMatrix(memCellCt, concatLen, -0.1, 0.1);
		
		// Biases terms
		/* this.bg = Matrix.random(memCellCt, 1);
		this.bi = Matrix.random(memCellCt, 1);
		this.bf = Matrix.random(memCellCt, 1);
		this.bo = Matrix.random(memCellCt, 1); */
		
		this.bg = NeuralNetUtils.initRandomMatrix(memCellCt, 1, -0.1, 0.1);
		this.bi = NeuralNetUtils.initRandomMatrix(memCellCt, 1, -0.1, 0.1);
		this.bf = NeuralNetUtils.initRandomMatrix(memCellCt, 1, -0.1, 0.1);
		this.bo = NeuralNetUtils.initRandomMatrix(memCellCt, 1, -0.1, 0.1);
		
		// Diffs (derivative of loss function w.r.t. all parameters)
		this.dWg = new Matrix(memCellCt, concatLen, 0.0);
		this.dWi = new Matrix(memCellCt, concatLen, 0.0);
		this.dWf = new Matrix(memCellCt, concatLen, 0.0);
		this.dWo = new Matrix(memCellCt, concatLen, 0.0);
		this.dbg = new Matrix(memCellCt, 1, 0.0);
		this.dbi = new Matrix(memCellCt, 1, 0.0);
		this.dbf = new Matrix(memCellCt, 1, 0.0);
		this.dbo = new Matrix(memCellCt, 1, 0.0);
		
		// Memory variables for AdaGrad.
		this.mWg = new Matrix(memCellCt, concatLen, 0.0);
		this.mWi = new Matrix(memCellCt, concatLen, 0.0);
		this.mWf = new Matrix(memCellCt, concatLen, 0.0);
		this.mWo = new Matrix(memCellCt, concatLen, 0.0);
		this.mbg = new Matrix(memCellCt, 1, 0.0);
		this.mbi = new Matrix(memCellCt, 1, 0.0);
		this.mbf = new Matrix(memCellCt, 1, 0.0);
		this.mbo = new Matrix(memCellCt, 1, 0.0);
	}
	
	public void gradientDescent() {
		if (this.useLRDecay) {
			if (decaySteps > 0) {
				this.learningRate = decayLearningRatePerStep(this.learningRate, this.learningRateDecayFactor, this.globalStep, this.decaySteps, this.staircase);
			} else {
				this.learningRate = decayLearningRate(this.learningRate, this.learningRateDecayFactor);
			}
		}
		
		this.Wg.plusEquals(this.dWg.times(this.learningRate).uminus());
		this.Wi.plusEquals(this.dWi.times(this.learningRate).uminus());
		this.Wf.plusEquals(this.dWf.times(this.learningRate).uminus());
		this.Wo.plusEquals(this.dWo.times(this.learningRate).uminus());
		this.bg.plusEquals(this.dbg.times(this.learningRate).uminus());
		this.bi.plusEquals(this.dbi.times(this.learningRate).uminus());
		this.bf.plusEquals(this.dbf.times(this.learningRate).uminus());
		this.bo.plusEquals(this.dbo.times(this.learningRate).uminus());
		
		// Reset diffs to zero.
		this.dWg = new Matrix(this.Wg.getRowDimension(), this.Wg.getColumnDimension(), 0.0);
		this.dWi = new Matrix(this.Wi.getRowDimension(), this.Wi.getColumnDimension(), 0.0);
		this.dWf = new Matrix(this.Wf.getRowDimension(), this.Wf.getColumnDimension(), 0.0);
		this.dWo = new Matrix(this.Wo.getRowDimension(), this.Wo.getColumnDimension(), 0.0);
		this.dbg = new Matrix(this.bg.getRowDimension(), this.bg.getColumnDimension(), 0.0);
		this.dbi = new Matrix(this.bi.getRowDimension(), this.bi.getColumnDimension(), 0.0);
		this.dbf = new Matrix(this.bf.getRowDimension(), this.bf.getColumnDimension(), 0.0);
		this.dbo = new Matrix(this.bo.getRowDimension(), this.bo.getColumnDimension(), 0.0);
	}
	
	public void adaGrad() {
		if (this.useLRDecay) {
			if (decaySteps > 0) {
				this.learningRate = decayLearningRatePerStep(this.learningRate, this.learningRateDecayFactor, this.globalStep, this.decaySteps, this.staircase);
			} else {
				this.learningRate = decayLearningRate(this.learningRate, this.learningRateDecayFactor);
			}
		}
		
		this.mWg.plusEquals(this.dWg.arrayTimes(this.dWg));
		this.mWi.plusEquals(this.dWi.arrayTimes(this.dWi));
		this.mWf.plusEquals(this.dWf.arrayTimes(this.dWf));
		this.mWo.plusEquals(this.dWo.arrayTimes(this.dWo));
		this.mbg.plusEquals(this.dbg.arrayTimes(this.dbg));
		this.mbi.plusEquals(this.dbi.arrayTimes(this.dbi));
		this.mbf.plusEquals(this.dbf.arrayTimes(this.dbf));
		this.mbo.plusEquals(this.dbo.arrayTimes(this.dbo));
		
		// Adagrad update.
		Matrix epsilonMat = new Matrix(this.Wg.getRowDimension(), this.Wg.getColumnDimension(), 1e-8);
		this.Wg.plusEquals(this.dWg.arrayRightDivide(MatrixUtils.sqrt(this.mWg.plus(epsilonMat))).times(this.learningRate).uminus());
		// Wxh.plusEquals(MatrixUtils.sqrt(mWxh.plus(epsilonMat)).arrayLeftDivide(dWxh).times(-learningRate));
		
		epsilonMat = new Matrix(this.Wi.getRowDimension(), this.Wi.getColumnDimension(), 1e-8);
		this.Wi.plusEquals(this.dWi.arrayRightDivide(MatrixUtils.sqrt(this.mWi.plus(epsilonMat))).times(this.learningRate).uminus());
		
		epsilonMat = new Matrix(this.Wf.getRowDimension(), this.Wf.getColumnDimension(), 1e-8);
		this.Wf.plusEquals(this.dWf.arrayRightDivide(MatrixUtils.sqrt(this.mWf.plus(epsilonMat))).times(this.learningRate).uminus());
		
		epsilonMat = new Matrix(this.Wo.getRowDimension(), this.Wo.getColumnDimension(), 1e-8);
		this.Wo.plusEquals(this.dWo.arrayRightDivide(MatrixUtils.sqrt(this.mWo.plus(epsilonMat))).times(this.learningRate).uminus());
		
		epsilonMat = new Matrix(this.bg.getRowDimension(), this.bg.getColumnDimension(), 1e-8);
		this.bg.plusEquals(this.dbg.arrayRightDivide(MatrixUtils.sqrt(this.mbg.plus(epsilonMat))).times(this.learningRate).uminus());
		
		epsilonMat = new Matrix(this.bi.getRowDimension(), this.bi.getColumnDimension(), 1e-8);
		this.bi.plusEquals(this.dbi.arrayRightDivide(MatrixUtils.sqrt(this.mbi.plus(epsilonMat))).times(this.learningRate).uminus());
		
		epsilonMat = new Matrix(this.bf.getRowDimension(), this.bf.getColumnDimension(), 1e-8);
		this.bf.plusEquals(this.dbf.arrayRightDivide(MatrixUtils.sqrt(this.mbf.plus(epsilonMat))).times(this.learningRate).uminus());
		
		epsilonMat = new Matrix(this.bo.getRowDimension(), this.bo.getColumnDimension(), 1e-8);
		this.bo.plusEquals(this.dbo.arrayRightDivide(MatrixUtils.sqrt(this.mbo.plus(epsilonMat))).times(this.learningRate).uminus());
		
		// Reset diffs to zero.
		this.dWg = new Matrix(this.Wg.getRowDimension(), this.Wg.getColumnDimension(), 0.0);
		this.dWi = new Matrix(this.Wi.getRowDimension(), this.Wi.getColumnDimension(), 0.0);
		this.dWf = new Matrix(this.Wf.getRowDimension(), this.Wf.getColumnDimension(), 0.0);
		this.dWo = new Matrix(this.Wo.getRowDimension(), this.Wo.getColumnDimension(), 0.0);
		this.dbg = new Matrix(this.bg.getRowDimension(), this.bg.getColumnDimension(), 0.0);
		this.dbi = new Matrix(this.bi.getRowDimension(), this.bi.getColumnDimension(), 0.0);
		this.dbf = new Matrix(this.bf.getRowDimension(), this.bf.getColumnDimension(), 0.0);
		this.dbo = new Matrix(this.bo.getRowDimension(), this.bo.getColumnDimension(), 0.0);
	}
	
	public double decayLearningRatePerStep(double lr, double decayFactor, int globalStep, int decaySteps, boolean staircase) {
		double decayedLr = 0.0;
		
		if (staircase) {
			decayedLr = lr * (Math.pow(decayFactor, (int)Math.floor(globalStep/decaySteps)));
		} else {
			decayedLr = lr * (Math.pow(decayFactor, (globalStep/decaySteps)));
		}
		
		return decayedLr;
	}
	
	public double decayLearningRate(double lr, double decayFactor) {
		return lr * decayFactor;
	}
	
	
}
