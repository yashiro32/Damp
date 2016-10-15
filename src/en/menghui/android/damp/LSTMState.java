package en.menghui.android.damp;

import Jama.Matrix;

public class LSTMState {
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
	
	public LSTMState(int nIn, int memCellCt) {
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
}
