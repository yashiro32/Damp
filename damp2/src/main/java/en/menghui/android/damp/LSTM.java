package en.menghui.android.damp;

import en.menghui.android.damp.utils.NeuralNetUtils;
import Jama.Matrix;

public class LSTM {
	public String type = "";
	
	public int nIn;
	public int nOut;
	public int nHidden = 0;
	public int memCellCt = 0;
	
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
	
	public Matrix g;
	public Matrix i;
	public Matrix f;
	public Matrix o;
	public Matrix s;
	public Matrix h;
	public Matrix bdh;
	public Matrix bds;
	public Matrix bdx;
	
	public Matrix hPrev;
	public Matrix sPrev;
	
	public Matrix x;
	public Matrix xc;
	
	public String activationFunction;
	
	public double regLambda = 0.01;
	public double learningRate = 0.1;
	
	public double learningRateDecayFactor = 1.0;
	public boolean useLRDecay = false;
	public boolean staircase = false;
	
	public int globalStep = 0;
	public int decaySteps = 0;
	
	public double dropoutP;
	
	public LSTM(int nIn, int nOut, String actFunc, double dropoutP, int nHidden, int memCellCt) {
		this.type = "lstm";
		
		this.nIn = nIn;
		this.nOut = nOut;
		this.nHidden = nHidden;
		this.activationFunction = actFunc;
		this.dropoutP = dropoutP;
		this.memCellCt = memCellCt;
		
		int concatLen = nIn + memCellCt;
		
		// Weight Matrixes.
		// this.Wg = Matrix.random(memCellCt, concatLen);
		// this.Wi = Matrix.random(memCellCt, concatLen);
		// this.Wf = Matrix.random(memCellCt, concatLen);
		// this.Wo = Matrix.random(memCellCt, concatLen);
		
		this.Wg = NeuralNetUtils.initRandomMatrix(memCellCt, concatLen, -0.1, 0.1);
		this.Wi = NeuralNetUtils.initRandomMatrix(memCellCt, concatLen, -0.1, 0.1);
		this.Wf = NeuralNetUtils.initRandomMatrix(memCellCt, concatLen, -0.1, 0.1);
		this.Wo = NeuralNetUtils.initRandomMatrix(memCellCt, concatLen, -0.1, 0.1);
		
		// Biases terms
		// this.bg = Matrix.random(memCellCt, 1);
		// this.bi = Matrix.random(memCellCt, 1);
		// this.bf = Matrix.random(memCellCt, 1);
		// this.bo = Matrix.random(memCellCt, 1);
		
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
		
		// LSTM States
		this.g = new Matrix(memCellCt, 1, 0.0);
		this.i = new Matrix(memCellCt, 1, 0.0);
		this.f = new Matrix(memCellCt, 1, 0.0);
		this.o = new Matrix(memCellCt, 1, 0.0);
		this.s = new Matrix(memCellCt, 1, 0.0);
		this.h = new Matrix(memCellCt, 1, 0.0);
		this.bdh = new Matrix(this.h.getRowDimension(), this.h.getColumnDimension());
		this.bds = new Matrix(this.s.getRowDimension(), this.s.getColumnDimension());
		this.bdx = new Matrix(nIn, 1, 0.0);
	}
	
	public void forwardProp(Matrix x, Matrix sPrev, Matrix hPrev) {
		if (sPrev == null) {
			sPrev = new Matrix(this.s.getRowDimension(), this.s.getColumnDimension());
		}
		
		if (hPrev == null) {
			hPrev = new Matrix(this.h.getRowDimension(), this.h.getColumnDimension());
		}
		
		this.sPrev = sPrev;
		this.hPrev = hPrev;
		
		// Concatenate x(t) and h(t - 1)
		Matrix xc = NeuralNetUtils.combineMatrixHorizontal(x.transpose(), hPrev.transpose()).transpose();
		
		this.g = NeuralNetUtils.tanh(NeuralNetUtils.add(this.Wg.times(xc), this.bg), false);
		this.i = NeuralNetUtils.sigmoid(NeuralNetUtils.add(this.Wi.times(xc), this.bi), false);
		this.f = NeuralNetUtils.sigmoid(NeuralNetUtils.add(this.Wf.times(xc), this.bf), false);
		this.o = NeuralNetUtils.sigmoid(NeuralNetUtils.add(this.Wo.times(xc), this.bo), false);
		this.s = NeuralNetUtils.add(g.arrayTimes(i), sPrev.arrayTimes(f));
		this.h = s.arrayTimes(o);
		
		this.x = x;
		this.xc = xc;
	}
	
	public void backProp(Matrix topDiffH, Matrix topDiffS) {
		// Notice that topDiffS is carried along the constant error carousel.
		// Matrix dds = this.o.arrayTimes(NeuralNetUtils.add(topDiffH, topDiffS));
		Matrix dds = NeuralNetUtils.add(this.o.arrayTimes(topDiffH), topDiffS);
		Matrix ddo = this.s.arrayTimes(topDiffH);
		Matrix ddi = this.g.arrayTimes(dds);
		Matrix ddg = this.i.arrayTimes(dds);
		Matrix ddf = this.sPrev.arrayTimes(dds);
		
		// Diffs w.r.t vector inside sigma / tanh function
		Matrix diInput = (new Matrix(i.getRowDimension(), i.getColumnDimension(), 1.0).plus(i.uminus())).arrayTimes(i).arrayTimes(ddi);
		Matrix dfInput = (new Matrix(f.getRowDimension(), f.getColumnDimension(), 1.0).plus(f.uminus())).arrayTimes(f).arrayTimes(ddf);
		Matrix doInput = (new Matrix(o.getRowDimension(), o.getColumnDimension(), 1.0).plus(o.uminus())).arrayTimes(o).arrayTimes(ddo);
		Matrix dgInput = (new Matrix(g.getRowDimension(), g.getColumnDimension(), 1.0).plus(g.arrayTimes(g).uminus())).arrayTimes(ddg);
		
		// Diffs w.r.t inputs
		this.dWi.plusEquals(diInput.times(this.xc.transpose()));
		this.dWf.plusEquals(dfInput.times(this.xc.transpose()));
		this.dWo.plusEquals(doInput.times(this.xc.transpose()));
		this.dWg.plusEquals(dgInput.times(this.xc.transpose()));
		this.dbi.plusEquals(diInput);
		this.dbf.plusEquals(dfInput);
		this.dbo.plusEquals(doInput);
		this.dbg.plusEquals(dgInput);
		
		// Compute bottom diff.
		Matrix dxc = new Matrix(this.xc.getRowDimension(), this.xc.getColumnDimension(), 0.0);
		dxc.plusEquals(this.Wi.transpose().times(diInput));
		dxc.plusEquals(this.Wf.transpose().times(dfInput));
		dxc.plusEquals(this.Wo.transpose().times(doInput));
		dxc.plusEquals(this.Wg.transpose().times(dgInput));
		
		// Save bottom diffs
		this.bds = dds.arrayTimes(this.f);
		this.bdx = dxc.getMatrix(0, this.nIn-1, 0, dxc.getColumnDimension()-1);
		this.bdh = dxc.getMatrix(this.nIn, dxc.getRowDimension()-1, 0, dxc.getColumnDimension()-1);
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
	
	public Matrix bottomDiff(Matrix pred, double label) {
		Matrix diff = new Matrix (pred.getRowDimension(), pred.getColumnDimension(), 0.0);
		
		diff.set(0, 0, 2.0 * (pred.get(0, 0) - label));
		
		return diff;
	}
	
	
}
