package en.menghui.android.damp;

import Jama.Matrix;

public class LstmCache {
	public Matrix WLSTM;
	public Matrix[] hOut;
	public Matrix[] ifogf;
	public Matrix[] ifog;
	public Matrix[] c;
	public Matrix[] ct;
	public Matrix[] hIn;
	public Matrix c0;
	public Matrix h0;
	
	public Matrix cPrev;
	public Matrix hPrev;
}
