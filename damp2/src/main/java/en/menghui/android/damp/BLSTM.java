package en.menghui.android.damp;

import javax.xml.xpath.XPath;

import en.menghui.android.damp.utils.MatrixUtils;
import en.menghui.android.damp.utils.NeuralNetUtils;
import Jama.Matrix;

public class BLSTM {
	public Matrix[] dX;
	public Matrix dWLSTM;
	public Matrix dc0;
	public Matrix dh0;
	
	public static Matrix init(int inputSize, int hiddenSize, double fancyForgetBiasInit) {
		// Initialize parameters of the LSTM (both weights and biases in one matrix).
		// One way might to have a positive fancy_forget_bias_init number(e.g. maybe even up to 5, in some papers).
		
		// +1 for the biases, which will be the first row of WLSTM.
		Matrix WLSTM = Matrix.random(inputSize+hiddenSize + 1, 4 * hiddenSize);
		WLSTM = WLSTM.times(1.0 / Math.sqrt(inputSize + hiddenSize)); // Matrix WLSTM divide by square root of (input_size + hidden_size).
		
		// Initialize biases to zero.
		Matrix zeroMat = new Matrix(1, WLSTM.getColumnDimension(), 0.0);
		WLSTM.setMatrix(0, 0, 0, WLSTM.getColumnDimension()-1, zeroMat);
		
		if (fancyForgetBiasInit != 0) {
			Matrix mat = new Matrix(1, hiddenSize, fancyForgetBiasInit);
			WLSTM.setMatrix(0, 0, hiddenSize, (2*hiddenSize)-1, mat);
		}
		
		return WLSTM;
	}
	
	public LstmCache forwardProp(Matrix[] X, Matrix WLSTM, Matrix c0, Matrix h0) {
		// X should be of shape (n,b,input_size), where n = length of sequence, b = batch size.
		int n = X.length;
		int b = X[0].getRowDimension();
		int inputSize = X[0].getColumnDimension();
		
		int d = WLSTM.getColumnDimension() / 4; // Hidden size.
		
		if (c0 == null) {
			c0 = new Matrix(b, d, 0.0);
		}
		
		if (h0 == null) {
			h0 = new Matrix(b, d, 0.0);
		}
		
		// Perform the LSTM forward pass with X as the input.
		int xphpb = WLSTM.getRowDimension(); // x plus h plus bias.
		
		Matrix[] hIn = createMatrixArray(n, b, xphpb, 0.0); // input [1, xt, ht-1] to each tick of the LSTM.
		Matrix[] hOut = createMatrixArray(n, b, d, 0.0); // Hidden representation of the LSTM (gated cell content).
		Matrix[] ifog = createMatrixArray(n, b, d * 4, 0.0); // input, forget, output, gate (IFOG).
		Matrix[] ifogf = createMatrixArray(n, b, d * 4, 0.0); // after nonlinearity.
		Matrix[] c = createMatrixArray(n, b, d, 0.0); // Cell content.
		Matrix[] ct = createMatrixArray(n, b, d, 0.0); // tanh of cell content.
		
		
		int t = 0; 
		while (t < n) {
			// Concat [x,h] as input to the LSTM
			Matrix prevH;
			if (t > 0)
				prevH = hOut[t-1];
			else
				prevH = h0;
			
			Matrix mat = new Matrix(hIn[t].getRowDimension(), 1, 1.0);
			hIn[t].setMatrix(0, hIn[t].getRowDimension()-1, 0, 0, mat); // bias
			
			hIn[t].setMatrix(0, hIn[t].getRowDimension()-1, 1, inputSize, X[t]);
			hIn[t].setMatrix(0, hIn[t].getRowDimension()-1, inputSize+1, hIn[t].getColumnDimension()-1, prevH);
			
			// Compute all gate activations. dots: (most work is this line)
			ifog[t] = hIn[t].times(WLSTM);
			// Non-linearities
			mat = NeuralNetUtils.sigmoid(ifogf[t].getMatrix(0, ifogf[t].getRowDimension()-1, 0, (3*d)-1), false);
			ifogf[t].setMatrix(0, ifogf[t].getRowDimension()-1, 0, (3*d)-1, mat);
			
			mat =  NeuralNetUtils.tanh(ifogf[t].getMatrix(0, ifogf[t].getRowDimension()-1, (3*d), ifogf[t].getColumnDimension()-1), false);
			ifogf[t].setMatrix(0, ifogf[t].getRowDimension()-1, (3*d), ifogf[t].getColumnDimension()-1, mat);
			
			// Compute the cell activation.
			Matrix prevC;
			if (t > 0)
				prevC = c[t-1];
			else
				prevC = c0;
			
			Matrix c1 = ifogf[t].getMatrix(0, ifogf[t].getRowDimension()-1, 0, d-1);
			Matrix c2 = ifogf[t].getMatrix(0, ifogf[t].getRowDimension()-1, (3*d), ifogf[t].getColumnDimension()-1);
			Matrix c3 = ifogf[t].getMatrix(0, ifogf[t].getRowDimension()-1, d, (2*d)-1);
			c[t] = NeuralNetUtils.add(c1.arrayTimes(c2), c3.arrayTimes(prevC));
			ct[t] = NeuralNetUtils.tanh(c[t], false);
			
			hOut[t] = ifogf[t].getMatrix(0, ifogf[t].getRowDimension()-1, 2*d, (3*d)-1).arrayTimes(ct[t]);
			
			t++;
		}
		
		LstmCache cache = new LstmCache();
		cache.WLSTM = WLSTM;
		cache.hOut = hOut;
		cache.ifogf = ifogf;
		cache.ifog = ifog;
		cache.c = c;
		cache.ct = ct;
		cache.hIn = hIn;
		cache.c0 = c0;
		cache.h0 = h0;
		
		cache.cPrev = c[t-1];
		cache.hPrev = hOut[t-1];
		
		return cache;
		
	}
	
	public void backProp(Matrix[] dHoutIn, LstmCache cache, Matrix dcn, Matrix dhn) {
		Matrix WLSTM = cache.WLSTM;
		Matrix[] hOut = cache.hOut;
		Matrix[] ifogf = cache.ifogf;
		Matrix[] ifog = cache.ifog;
		Matrix[] c = cache.c;
		Matrix[] ct = cache.ct;
		Matrix[] hIn = cache.hIn;
		Matrix c0 = cache.c0;
		Matrix h0 = cache.h0;
		
		int n = hOut.length;
		int b = hOut[0].getRowDimension();
		int d = hOut[0].getColumnDimension();
		int inputSize = WLSTM.getRowDimension() - d - 1; // -1 due to bias.
		
		// Backprop the LSTM.
		Matrix[] dIfog = createMatrixArray(ifog.length, ifog[0].getRowDimension(), ifog[0].getColumnDimension(), 0.0);
		Matrix[] dIfogf = createMatrixArray(ifogf.length, ifogf[0].getRowDimension(), ifogf[0].getColumnDimension(), 0.0);
		Matrix dWLSTM = new Matrix(WLSTM.getRowDimension(), WLSTM.getColumnDimension(), 0.0);
		Matrix[] dHin = createMatrixArray(hIn.length, hIn[0].getRowDimension(), hIn[0].getColumnDimension(), 0.0);
		Matrix[] dC = createMatrixArray(c.length, c[0].getRowDimension(), c[0].getColumnDimension(), 0.0);
		Matrix[] dX = createMatrixArray(n, b, inputSize, 0.0);
		Matrix dh0 = new Matrix(b, d, 0.0);
		Matrix dc0 = new Matrix(b, d, 0.0);
		Matrix[] dHout = MatrixUtils.copyMatrixArray(dHoutIn); // Make a copy so we don't have any side effects.
		
		if (dcn != null) {
			dC[n-1].plusEquals(dcn.copy());
		}
		
		if (dhn != null) {
			dHout[n-1].plusEquals(dhn.copy());
		}
		
		for (int t = 0; t < n; t++) {
			Matrix tanhCt = ct[t];
			dIfogf[t].setMatrix(0, dIfogf[t].getRowDimension()-1, 2*d, (3*d)-1, tanhCt.arrayTimes(dHout[t]));
			
			// Backprop tanh non-linearity first then continue backprop
			Matrix c1 = ifogf[t].getMatrix(0, dIfogf[t].getRowDimension()-1, 2*d, (3*d)-1);
			Matrix oneMat = new Matrix(tanhCt.getRowDimension(), tanhCt.getColumnDimension(), 1.0);
			Matrix c2 = oneMat.minus(tanhCt.arrayTimes(tanhCt));
			dC[t].plusEquals(c2.arrayTimes(c1).arrayTimes(dHout[t]));
			
			if (t > 0) {
				dIfogf[t].setMatrix(0, dIfogf[t].getRowDimension()-1, d, (2*d)-1, c[t-1].arrayTimes(dC[t]));
				Matrix mat = ifogf[t].getMatrix(0, ifogf[t].getRowDimension()-1, d, (2*d)-1);
				dC[t-1].plusEquals(mat.arrayTimes(dC[t]));
			} else {
				dIfogf[t].setMatrix(0, dIfogf[t].getRowDimension()-1, d, (2*d)-1, c0.arrayTimes(dC[t]));
				dc0 = ifogf[t].getMatrix(0, ifogf[t].getRowDimension()-1, d, (2*d)-1).arrayTimes(dC[t]);
			}
			
			c1 = ifogf[t].getMatrix(0, ifogf[t].getRowDimension()-1, 3*d, ifogf[t].getColumnDimension()-1);
			c2 = ifogf[t].getMatrix(0, ifogf[t].getRowDimension()-1, 0, d-1);
			dIfogf[t].setMatrix(0, dIfogf[t].getRowDimension()-1, 0, d-1, c1.arrayTimes(dC[t]));
			dIfogf[t].setMatrix(0, dIfogf[t].getRowDimension()-1, 3*d, dIfogf[t].getColumnDimension()-1, c2.arrayTimes(dC[t]));
			
			// Backprop activation functions.
			c1 = ifogf[t].getMatrix(0, ifogf[t].getRowDimension()-1, 3*d, ifogf[t].getColumnDimension()-1);
			c2 = dIfogf[t].getMatrix(0, dIfogf[t].getRowDimension()-1, 3*d, dIfogf[t].getColumnDimension()-1);
			oneMat = new Matrix(c1.getRowDimension(), c1.getColumnDimension(), 1.0);
			dIfog[t].setMatrix(0, dIfogf[t].getRowDimension()-1, 3*d, dIfogf[t].getColumnDimension()-1, oneMat.minus(c1.arrayTimes(c1)).arrayTimes(c2));
			
			c1 = ifogf[t].getMatrix(0, ifogf[t].getRowDimension()-1, 0, (3*d)-1);
			c2 = dIfogf[t].getMatrix(0, dIfogf[t].getRowDimension()-1, 0, (3*d)-1);
			oneMat = new Matrix(c1.getRowDimension(), c1.getColumnDimension(), 1.0);
			dIfog[t].setMatrix(0, dIfog[t].getRowDimension()-1, 0, (3*d)-1, c1.arrayTimes(oneMat.minus(c1)).arrayTimes(c2));
			
			// Backprop matrix multiply.
			dWLSTM.plusEquals(hIn[t].transpose().times(dIfog[t]));
			dHin[t] = dIfog[t].times(WLSTM.transpose());
			
			// Backprop the identity transform into Hin.
			dX[t] = dHin[t].getMatrix(0, dHin[t].getRowDimension()-1, 1, inputSize);
			if (t > 0) {
				dHout[t-1] = dHin[t].getMatrix(0, dHin[t].getRowDimension()-1, inputSize+1, dHin[t].getColumnDimension()-1);
			} else {
				dh0.plusEquals(dHin[t].getMatrix(0, dHin[t].getRowDimension()-1, inputSize+1, dHin[t].getColumnDimension()-1));
			}
		}
		
		this.dX = dX;
		this.dWLSTM = dWLSTM;
		this.dc0 = dc0;
		this.dh0 = dh0;
	}
	
	private Matrix[] createMatrixArray(int n, int row, int col, double val) {
		Matrix[] matArr = new Matrix[n];
		for (int i = 0; i < matArr.length; i++) {
			Matrix mat = new Matrix(row, col, val);
			matArr[i] = mat;
		}
		
		return matArr;
	}
}
