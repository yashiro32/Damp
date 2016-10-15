package en.menghui.android.damp;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import en.menghui.android.damp.recurrent.lstm.LSTMNetwork;
import en.menghui.android.damp.recurrent.lstm.LSTMNode;
import en.menghui.android.damp.recurrent.lstm.LSTMParam;
import en.menghui.android.damp.recurrent.lstm.LSTMState;
import Jama.Matrix;
import android.app.Activity;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;

public class MainActivity extends Activity {

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		
		// simpleImplementation();
		// oneHiddenLayerImplementation();
		
		double[][] x = {{0.0,0.0,1.0}, {0.0,1.0,1.0}, {1.0,0.0,1.0}, {1.0,1.0,1.0}};
		double[] ys = {0.0,0.0,1.0,1.0};
		Matrix X = new Matrix(x);
		Matrix y = new Matrix(ys, 4);
		
		// Network net1 = new Network(X, y);
		// net1.forwardProp();
		
		FullyConnectedLayer fc1 = new FullyConnectedLayer(3, 10, "sigmoid", 0.0);
		// FullyConnectedLayer fc2 = new FullyConnectedLayer(10, 15, "sigmoid", 0.0);
		SoftmaxLayer sf1 = new SoftmaxLayer(10, 2, 0.0);
		
		for (int i = 0; i < 0; i++) {
			fc1.setInput(X, X, 60);
			// fc2.setInput(fc1.output, fc1.output, 60);
			sf1.setInput(fc1.output, fc1.output, y, 60);
			
			sf1.backProp(null);
			// fc2.backProp(sf1.bpOutput);
			fc1.backProp(sf1.bpOutput);
			
			// NeuralNetUtils.printMatrix(fc1.output);
		}
		
		// NeuralNetUtils.printMatrix(sf1.output);
		// NeuralNetUtils.printMatrix(sf1.yOut);
		
		// testConv();
		// testPool();
		// testRecurrent();
		testLSTM(0.1);
		// testBLSTM();
	}
	
	private void simpleImplementation() {
		double[][] x = {{0,0,1}, {0,1,1}, {1,0,1}, {1,1,1}};
		double[] ys = {0,0,1,1};
		Matrix X = new Matrix(x);
		Matrix y = new Matrix(ys, 4);
		
		Matrix oneMat = new Matrix(3, 1, 1.0);
		Matrix w = Matrix.random(3, 1).times(2).minus(oneMat);
		
		Matrix l1 = null;
		
		for (int i = 0; i < 100; i++) {
			// Forward propagation 
			Matrix l0 = X;
			l1 = NeuralNetUtils.sigmoid(l0.times(w), false);
			
			// How much did we miss?
			Matrix l1Error = y.minus(l1);
			
			// Multiply how much we missed by the slope of the sigmoid ar the values in l1
			Matrix l1Delta = l1Error.arrayTimes(NeuralNetUtils.sigmoid(l1, true));
			
			// Update weights
			w.plusEquals(l0.transpose().times(l1Delta));
		}
		
		NeuralNetUtils.printMatrix(l1);
	}
	
	private void oneHiddenLayerImplementation() {
		double[][] x = {{0.0,0.0,1.0}, {0.0,1.0,1.0}, {1.0,0.0,1.0}, {1.0,1.0,1.0}};
		double[] ys = {0.0,0.0,1.0,1.0};
		Matrix X = new Matrix(x);
		Matrix y = new Matrix(ys, 4);
		
		Matrix oneMat = new Matrix(3, 10, 1.0);
		Matrix W1 = Matrix.random(3, 10);
		Matrix b1 = new Matrix(1, 10, 0.0);
		oneMat = new Matrix(10, 2, 1.0);
		Matrix W2 = Matrix.random(10, 2);
		Matrix b2 = new Matrix(1, 2, 0.0);
		
		/* double[][] w1 = {{1.0184761, 0.23103087, 0.56507464}, {1.29378029, 1.07823511, -0.56423165}, {0.5485338, -0.08738612, -0.05959343}};
		Matrix W1 = new Matrix(w1);
		double[][] w2 = {{0.23705916, 0.08316359}, {0.8396252, 0.43938534}, {0.0702491, 0.25626456}};
		Matrix W2 = new Matrix(w2); */
		
		for (int j = 0; j < 1000; j++) {
			// Forward propagation
			Matrix z1 = NeuralNetUtils.add(X.times(W1), b1);
			Matrix a1 = NeuralNetUtils.sigmoid(z1, false);
			
			Matrix z2 = NeuralNetUtils.add(a1.times(W2), b2);
			Matrix exp = NeuralNetUtils.sigmoid(z2, false);
			
			NeuralNetUtils.printMatrix(exp);
			
			// Back Propagation
			// Matrix delta3 = y.minus(exp);
			Matrix delta3 = exp.copy();
			for (int k = 0; k < delta3.getRowDimension(); k++) {
				delta3.set(k, (int)y.get(k, 0), delta3.get(k, (int)y.get(k, 0)) - 1.0);
			}
			
			Matrix dW2 = a1.transpose().times(delta3);
			Matrix db2 = NeuralNetUtils.sum(delta3, 0);
			Matrix delta2 = delta3.times(W2.transpose()).arrayTimes(NeuralNetUtils.sigmoid(a1, true));
			
			Matrix dW1 = X.transpose().times(delta2);
			Matrix db1 = NeuralNetUtils.sum(delta2, 0);
			
			// NeuralNetUtils.printMatrix(dW1);
			// NeuralNetUtils.printMatrix(dW2);
			// NeuralNetUtils.printMatrix(exp);
			
		    // Add regularization terms (b1 and b2 don't have regularization terms)
			dW2.plusEquals(W2.times(0.01));
			dW1.plusEquals(W1.times(0.01));
			
			W1.plusEquals(dW1.times(-0.01));
			b1.plusEquals(db1.times(-0.01));
			W2.plusEquals(dW2.times(-0.01));
			b2.plusEquals(db2.times(-0.01));
		
		}
		
		/* double[][] dou = {{-1.25624274, -0.00535655}, {-1.38972751, 0.01815276}, {-1.22335244, 0.22192261}, {-1.33113312, 0.23956048}};
		Matrix mat = new Matrix(dou);
		Matrix sig = NeuralNetUtils.sigmoid(mat, false);
		NeuralNetUtils.printMatrix(mat);
		NeuralNetUtils.printMatrix(sig); */
	}
	
	private void testConv() {
		Matrix img = Matrix.identity(8, 8);
		Matrix[][] imgs = new Matrix[1][1];
		imgs[0][0] = img;
		Matrix filter = Matrix.identity(7, 7);
		Matrix[][] filters = new Matrix[1][1];
		filters[0][0] = filter;
		Matrix conv = new Matrix(8, 8);
		Matrix[][] convout = new Matrix[1][1];
		convout[0][0] = conv;
		
		ConvolutionLayer cl1 = new ConvolutionLayer(1, 1, 1, 1, 8, 8, 7, 7);
		cl1.filters = filters;
		cl1.setInput(imgs, 60);
		
		for (int i = 0; i < cl1.convOutputs.length; i++) {
			for (int j = 0; j < cl1.convOutputs[0].length; j++) {
				NeuralNetUtils.printMatrix(cl1.convOutputs[i][j]);
			}
		}
	}
	
	private void testPool() {
		int imageHeight = 3;
		int imageWidth = 3;
		int strideY = 1;
		int strideX = 1;
		int poolHeight = 2;
		int poolWidth = 2;
		
		int pooloutH = imageHeight / strideY;
		int pooloutW = imageWidth / strideX;
		
		Matrix img = Matrix.identity(imageHeight,  imageWidth);
		img.set(0, 1, 2.0);
		Matrix[][] imgs = new Matrix[1][1];
		imgs[0][0] = img;
		
		PoolingLayer pl1 = new PoolingLayer(1, 1, imageHeight, imageWidth, strideY, strideX, poolHeight, poolWidth);
		
		for (int i = 0; i < pl1.poolout.length; i++) {
			for (int j = 0; j < pl1.poolout[0].length; j++) {
				Matrix mat = new Matrix(pooloutH, pooloutW);
				pl1.poolout[i][j] = mat;
			}
		}
		
		pl1.setInput(imgs, 60);
		
		for (int i = 0; i < pl1.poolout.length; i++) {
			for (int j = 0; j < pl1.poolout[0].length; j++) {
				NeuralNetUtils.printMatrix(pl1.poolout[i][j]);
			}
		}
		
	}
	
	public void testRecurrent() {
		Matrix[] inputs = new Matrix[25];
		Matrix[] targets = new Matrix[25];
		Matrix hprev = new Matrix(100, 1, 0.0);
		
		inputs[0] = new Matrix(1, 1, 19);
		inputs[1] = new Matrix(1, 1, 46); 
		inputs[2] = new Matrix(1, 1, 57); 
		inputs[3] = new Matrix(1, 1, 56); 
		inputs[4] = new Matrix(1, 1, 59);
		inputs[5] = new Matrix(1, 1, 2);
		inputs[6] = new Matrix(1, 1, 14); 
		inputs[7] = new Matrix(1, 1, 46); 
		inputs[8] = new Matrix(1, 1, 59); 
		inputs[9] = new Matrix(1, 1, 46);
		inputs[10] = new Matrix(1, 1, 64);
		inputs[11] = new Matrix(1, 1, 42); 
		inputs[12] = new Matrix(1, 1, 53); 
		inputs[13] = new Matrix(1, 1, 11); 
		inputs[14] = new Matrix(1, 1, 0);
		inputs[15] = new Matrix(1, 1, 15);
		inputs[16] = new Matrix(1, 1, 42); 
		inputs[17] = new Matrix(1, 1, 45); 
		inputs[18] = new Matrix(1, 1, 52); 
		inputs[19] = new Matrix(1, 1, 57);
		inputs[20] = new Matrix(1, 1, 42);
		inputs[21] = new Matrix(1, 1, 2); 
		inputs[22] = new Matrix(1, 1, 60); 
		inputs[23] = new Matrix(1, 1, 42); 
		inputs[24] = new Matrix(1, 1, 2);
		
		targets[0] = new Matrix(1, 1, 46);
		targets[1] = new Matrix(1, 1, 57); 
		targets[2] = new Matrix(1, 1, 56); 
		targets[3] = new Matrix(1, 1, 59); 
		targets[4] = new Matrix(1, 1, 2);
		targets[5] = new Matrix(1, 1, 14);
		targets[6] = new Matrix(1, 1, 46); 
		targets[7] = new Matrix(1, 1, 59); 
		targets[8] = new Matrix(1, 1, 46); 
		targets[9] = new Matrix(1, 1, 64);
		targets[10] = new Matrix(1, 1, 42);
		targets[11] = new Matrix(1, 1, 53); 
		targets[12] = new Matrix(1, 1, 11); 
		targets[13] = new Matrix(1, 1, 0); 
		targets[14] = new Matrix(1, 1, 15);
		targets[15] = new Matrix(1, 1, 42);
		targets[16] = new Matrix(1, 1, 45); 
		targets[17] = new Matrix(1, 1, 52); 
		targets[18] = new Matrix(1, 1, 57); 
		targets[19] = new Matrix(1, 1, 42);
		targets[20] = new Matrix(1, 1, 2);
		targets[21] = new Matrix(1, 1, 60); 
		targets[22] = new Matrix(1, 1, 42); 
		targets[23] = new Matrix(1, 1, 2); 
		targets[24] = new Matrix(1, 1, 55); 
		
		RecurrentLayer layer = new RecurrentLayer(65, 65, "tanh", 0.5, 100);
		layer.setArgumentsForPropagation(inputs, targets, hprev);
		layer.forwardProp();
		layer.backProp();
		layer.adaGrad();
		
	}
	
	private void testLSTM() {
		int memCellCt = 100;
		int xDim = 50;
		
		int epoch = 100;
		
		LSTM layer = new LSTM(xDim, 60, "sigmoid", 0.5, 78, memCellCt);
		
		double[] yList = {-0.5, 0.2, 0.1, -0.5};
		 
		Matrix[] xList = new Matrix[yList.length];
		 
		LSTMState[] stateList = new LSTMState[yList.length];
		
		for (int i = 0; i < yList.length; i++) {
			Matrix xVal = new Matrix(xDim, 1);
			 
			for (int j = 0; j < xDim; j++) {
				xVal.set(j, 0, Math.random()); 
			}
			  
			xList[i] = xVal;
		}
		
		for (int iter = 0; iter < epoch; iter++) {
			// Forward Prop
			for (int i = 0; i < yList.length; i++) {
				LSTMState state = new LSTMState(xDim, memCellCt);
				stateList[i] = state;
				
				stateList[i].sPrev = layer.s;
				stateList[i].hPrev = layer.h;
				
				layer.forwardProp(xList[i], stateList[i].sPrev, stateList[i].hPrev);
				
				stateList[i].s = layer.s;
				stateList[i].h = layer.h;
				 
				stateList[i].bds = layer.bds;
				stateList[i].bdh = layer.bdh;
				
				System.out.println("y_pred[" + i + "] : " + layer.h.get(0, 0));
				
			}
			
			// Backward Prop
			int idx = xList.length - 1;
			
			Matrix diffH = layer.bottomDiff(stateList[idx].h, yList[idx]);
			Matrix diffS = new Matrix (memCellCt, 1, 0.0);
			layer.backProp(diffH, diffS);
			
			idx--;
			
			while (idx >= 0) {
				diffH = layer.bottomDiff(stateList[idx].h, yList[idx]);
				diffH.plusEquals(stateList[idx+1].bdh);
				diffS = stateList[idx+1].bds;
				
				layer.backProp(diffH, diffS);
				
				idx--;
			}
			
			// Gradient Descent.
			layer.gradientDescent();
			
		}
		
	}
	
	private void testLSTM(double lr) {
		int memCellCt = 100;
		int xDim = 50;
		int concatLen = xDim + memCellCt;
		LSTMParam lstmParam = new LSTMParam(memCellCt, xDim);
		lstmParam.learningRate = 0.05;
		// lstmParam.learningRateDecayFactor = 0.1;
		// lstmParam.useLRDecay = true;
		LSTMNetwork lstmNet = new LSTMNetwork(lstmParam);
		int epoch = 100;
		
		double[] yList = {-0.5, 0.2, 0.1, -0.5};
		
		Matrix[] xList = new Matrix[yList.length];
		
		for (int u = 0; u < yList.length; u++) {
			Matrix xVal = new Matrix(xDim, 1);
			 
			for (int j = 0; j < xDim; j++) {
				xVal.set(j, 0, Math.random()); 
			}
			  
			xList[u] = xVal;
		}
		
		for (int iter = 0; iter < epoch; iter++) {
			for (int i = 0; i < yList.length; i++) {
				lstmNet.xListAdd(xList[i]);
				System.out.println("y_pred[" + i + "] : " + lstmNet.lstmNodeList.get(i).state.h.get(0, 0));
			}
			
			lstmNet.yListIs(yList);
			lstmParam.adaGrad();
			lstmNet.xListClear();
		}
	}
	
	public void testBLSTM() {
		int n = 5; // Sequence length
		int b = 3; // batch size
		int d = 4; // hidden size
		int inputSize = 10;
		
		Matrix WLSTM = BLSTM.init(inputSize, d, 3.0); // input size, hidden size, forget bias.
		BLSTM layer = new BLSTM();
		
		Matrix[] X = new Matrix[n];
		for (int i = 0; i < n; i++) {
			Matrix mat = Matrix.random(b, inputSize);
			X[i] = mat;
		}
		
		Matrix h0 = Matrix.random(b, d);
		Matrix c0 = Matrix.random(b, d);
		
		// Sequential forward
		Matrix cPrev = c0;
		Matrix hPrev = h0;
		
		LstmCache[] caches = new LstmCache[n];
		Matrix[] hCat = MatrixUtils.createMatrixArray(n, b, d, 0.0);
		for (int t = 0; t < n; t++) {
			Matrix mat = X[t];
			Matrix[] xt = new Matrix[1];
			xt[0] = mat;
			LstmCache cache = layer.forwardProp(xt, WLSTM, cPrev, hPrev);
			caches[t] = cache;
			hCat[t] = hPrev;
		}
		
		Matrix[] dH = new Matrix[hCat.length];
		for (int i = 0; i < dH.length; i++) {
			dH[i] = Matrix.random(hCat[0].getRowDimension(), hCat[0].getColumnDimension());
		}
		
		// Perform sequential gradients.
		Matrix[] dX = MatrixUtils.createMatrixArray(X.length, X[0].getRowDimension(), X[0].getColumnDimension(), 0.0);
		Matrix dWLSTM = new Matrix(WLSTM.getRowDimension(), WLSTM.getColumnDimension());
		Matrix dc0 = new Matrix(c0.getRowDimension(), c0.getColumnDimension());
		Matrix dh0 = new Matrix(h0.getRowDimension(), h0.getColumnDimension());
		Matrix dcNext = null;
		Matrix dhNext = null;
		
		for (int t = n-1; t > -1; t--) {
			Matrix[] dht = new Matrix[1];
			dht[0] = dH[t];
			
			layer.backProp(dht, caches[t], dcNext, dhNext);
			dhNext = layer.dh0;
			dcNext = layer.dc0;
			
			dWLSTM.plusEquals(layer.dWLSTM);
			dX[t] = layer.dX[0];
			
			if (t == 0) {
				dc0 = layer.dc0;
				dh0 = layer.dh0;
			}
		}
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		// Inflate the menu; this adds items to the action bar if it is present.
		getMenuInflater().inflate(R.menu.main, menu);
		return true;
	}
	
	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		// Handle action bar item clicks here. The action bar will
		// automatically handle clicks on the Home/Up button, so long
		// as you specify a parent activity in AndroidManifest.xml.
		int id = item.getItemId();
		/* if (id == R.id.action_settings) {
			return true;
		} */
		return super.onOptionsItemSelected(item);
	}
	
	
}
