package en.menghui.android.damp.recurrent.lstm;

import en.menghui.android.damp.utils.NeuralNetUtils;
import Jama.Matrix;

public class LSTMNode {
	public LSTMState state;
	public LSTMParam param;
	
	public Matrix hPrev;
	public Matrix sPrev;
	
	public Matrix x;
	public Matrix xc;
	
	public LSTMNode(LSTMParam lstmParam, LSTMState lstmState) {
		this.state = lstmState;
		this.param = lstmParam;
	}
	
	public void forwardProp(Matrix x, Matrix sPrev, Matrix hPrev) {
		if (sPrev == null) {
			sPrev = new Matrix(this.state.s.getRowDimension(), this.state.s.getColumnDimension());
		}
		
		if (hPrev == null) {
			hPrev = new Matrix(this.state.h.getRowDimension(), this.state.h.getColumnDimension());
		}
		
		this.sPrev = sPrev;
		this.hPrev = hPrev;
		
		// Concatenate x(t) and h(t - 1)
		Matrix xc = NeuralNetUtils.combineMatrixHorizontal(x.transpose(), hPrev.transpose()).transpose();
		
		this.state.g = NeuralNetUtils.tanh(NeuralNetUtils.add(this.param.Wg.times(xc), this.param.bg), false);
		this.state.i = NeuralNetUtils.sigmoid(NeuralNetUtils.add(this.param.Wi.times(xc), this.param.bi), false);
		this.state.f = NeuralNetUtils.sigmoid(NeuralNetUtils.add(this.param.Wf.times(xc), this.param.bf), false);
		this.state.o = NeuralNetUtils.sigmoid(NeuralNetUtils.add(this.param.Wo.times(xc), this.param.bo), false);
		this.state.s = NeuralNetUtils.add(this.state.g.arrayTimes(this.state.i), sPrev.arrayTimes(this.state.f));
		this.state.h = this.state.s.arrayTimes(this.state.o);
		
		this.x = x;
		this.xc = xc;
	}
	
	public void backProp(Matrix topDiffH, Matrix topDiffS) {
		// Notice that topDiffS is carried along the constant error carousel.
		// Matrix dds = this.state.o.arrayTimes(NeuralNetUtils.add(topDiffH, topDiffS));
		Matrix dds = NeuralNetUtils.add(this.state.o.arrayTimes(topDiffH), topDiffS);
		Matrix ddo = this.state.s.arrayTimes(topDiffH);
		Matrix ddi = this.state.g.arrayTimes(dds);
		Matrix ddg = this.state.i.arrayTimes(dds);
		Matrix ddf = this.sPrev.arrayTimes(dds);
		
		// Diffs w.r.t vector inside sigma / tanh function
		Matrix diInput = (new Matrix(this.state.i.getRowDimension(), this.state.i.getColumnDimension(), 1.0).plus(this.state.i.uminus())).arrayTimes(this.state.i).arrayTimes(ddi);
		Matrix dfInput = (new Matrix(this.state.f.getRowDimension(), this.state.f.getColumnDimension(), 1.0).plus(this.state.f.uminus())).arrayTimes(this.state.f).arrayTimes(ddf);
		Matrix doInput = (new Matrix(this.state.o.getRowDimension(), this.state.o.getColumnDimension(), 1.0).plus(this.state.o.uminus())).arrayTimes(this.state.o).arrayTimes(ddo);
		Matrix dgInput = (new Matrix(this.state.g.getRowDimension(), this.state.g.getColumnDimension(), 1.0).plus(this.state.g.arrayTimes(this.state.g).uminus())).arrayTimes(ddg);
		
		// Diffs w.r.t inputs
		this.param.dWi.plusEquals(diInput.times(this.xc.transpose()));
		this.param.dWf.plusEquals(dfInput.times(this.xc.transpose()));
		this.param.dWo.plusEquals(doInput.times(this.xc.transpose()));
		this.param.dWg.plusEquals(dgInput.times(this.xc.transpose()));
		this.param.dbi.plusEquals(diInput);
		this.param.dbf.plusEquals(dfInput);
		this.param.dbo.plusEquals(doInput);
		this.param.dbg.plusEquals(dgInput);
		
		// Compute bottom diff.
		Matrix dxc = new Matrix(this.xc.getRowDimension(), this.xc.getColumnDimension(), 0.0);
		dxc.plusEquals(this.param.Wi.transpose().times(diInput));
		dxc.plusEquals(this.param.Wf.transpose().times(dfInput));
		dxc.plusEquals(this.param.Wo.transpose().times(doInput));
		dxc.plusEquals(this.param.Wg.transpose().times(dgInput));
		
		// Save bottom diffs
		this.state.bds = dds.arrayTimes(this.state.f);
		this.state.bdx = dxc.getMatrix(0, this.param.xDim-1, 0, dxc.getColumnDimension()-1);
		this.state.bdh = dxc.getMatrix(this.param.xDim, dxc.getRowDimension()-1, 0, dxc.getColumnDimension()-1);
	}
}
