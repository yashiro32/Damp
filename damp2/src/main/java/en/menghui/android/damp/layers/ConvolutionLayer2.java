package en.menghui.android.damp.layers;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import android.util.Log;
import Jama.Matrix;
import en.menghui.android.damp.arrays.Tensor;
import en.menghui.android.damp.optimizations.AdaDeltaOptimizer;
import en.menghui.android.damp.optimizations.AdaGradOptimizer;
import en.menghui.android.damp.optimizations.AdamOptimizer;
import en.menghui.android.damp.optimizations.GDOptimizer;
import en.menghui.android.damp.optimizations.NetsterovOptimizer;
import en.menghui.android.damp.optimizations.SGDOptimizer;
import en.menghui.android.damp.optimizations.WindowGradOptimizer;
import en.menghui.android.damp.utils.MatrixUtils;
import en.menghui.android.damp.utils.NeuralNetUtils;

public class ConvolutionLayer2 extends Layer {
	private static final String TAG = "Convolution Layer 2";
	
	public String type = "";
	
	public int numImages;
	public int numFilters;
	public int numChannelsIn;
	public int numChannelsOut;
	public int imageWidth;
	public int imageHeight;
	public int filterWidth;
	public int filterHeight;
	public int filMidW;
	public int filMidH;
	
	public Tensor images;
	public Tensor filters;
	public Tensor convOutputs;
	
	public Tensor W;
	public Tensor b;
	
	public Tensor dW;
	public Tensor db;
	
	public Tensor mW;
	public Tensor mb;
	
	public Tensor vW;
	public Tensor vb;
	
	public Tensor input;
	public Tensor output;
	
	public Tensor bpInput;
	public Tensor bpOutput;
	
	public Tensor dropoutTensor;
	
	// Batch Normalization parameters.
	Matrix gamma;
	Matrix beta;
	Matrix xmu;
	Matrix var;
	Matrix sqrtvar;
	Matrix ivar;
	Matrix xhat;
	Matrix batchNormOut;
	
	public ConvolutionLayer2(int numImages, int numFilters, int numChannelsIn, int numChannelsOut, int imgWidth, int imgHeight, int filterWidth, int filterHeight) {
		this.type = "convolution";
		
		this.numImages = numImages;
		this.numFilters = numFilters;
		this.numChannelsIn = numChannelsIn;
		this.numChannelsOut = numChannelsOut;
		this.imageWidth = imgWidth;
		this.imageHeight = imgHeight;
		this.filterWidth = filterWidth;
		this.filterHeight = filterHeight;
		
		filMidW = filterWidth / 2;
		filMidH = filterHeight / 2;
		
		images = new Tensor(Arrays.asList(numImages, numChannelsIn, imgHeight, imgWidth));
		filters = new Tensor(Arrays.asList(numChannelsIn, numChannelsOut, filterHeight, filterWidth), true);
		convOutputs = new Tensor(Arrays.asList(numImages, numChannelsOut, imgHeight, imgWidth));
		
		this.W = new Tensor(Arrays.asList(numChannelsIn, numChannelsOut, filterHeight, filterWidth), true);
		this.b = new Tensor(Arrays.asList(1, numChannelsOut, 1, 1));
		
		this.dW = new Tensor(Arrays.asList(numChannelsIn, numChannelsOut, filterHeight, filterWidth));
		this.db = new Tensor(Arrays.asList(1, numChannelsOut, 1, 1));
		
		this.mW = new Tensor(Arrays.asList(numChannelsIn, numChannelsOut, filterHeight, filterWidth));
		this.mb = new Tensor(Arrays.asList(1, numChannelsOut, 1, 1));
		
		this.vW = new Tensor(Arrays.asList(numChannelsIn, numChannelsOut, filterHeight, filterWidth));
		this.vb = new Tensor(Arrays.asList(1, numChannelsOut, 1, 1));
		
		this.bpOutput = new Tensor(this.images.shape);
		
	}
	
	public void forwarProp(Tensor inpt, int miniBatchSize) {
		this.images = inpt;
		this.images.shape = inpt.shape;
		this.input = this.images.clone();
		
		for (int i = 0; i < numImages; i++) {
			for (int cOut = 0; cOut < numChannelsOut; cOut++) {
				for (int y = 0; y < imageHeight; y++) {
					int yOffMin = intMax(-y, -filMidH);
					int yOffMax	= intMin(imageHeight-y, filMidH+1);
					for (int x = 0; x < imageWidth; x++) {
						int xOffMin = intMax(-x, -filMidW);
						int xOffMax = intMin(imageWidth-x, filMidW+1);
						double value = 0.0;
						for (int yOff = yOffMin; yOff < yOffMax; yOff++) {
							for (int xOff = xOffMin; xOff < xOffMax; xOff++) {
								int imageY = y + yOff;
								int imageX = x + xOff;
								int filY = filMidW + yOff;
								int filX = filMidH + xOff;
								for (int cIn = 0; cIn < numChannelsIn; cIn++) {
									value += images.get(i, cIn, imageY, imageX) * this.W.get(cIn, cOut, filY, filX);
								}
							}
						}
						this.convOutputs.set(i, cOut, y, x, value);
					}
				}
			}
		}
		
		Tensor h = addTensor(this.convOutputs, this.b).clone();
		if (this.useBatchNormalization) {
			h.tmat = batchNormalizationForward(h.tmat);
		}
		
		this.output = new Tensor(this.convOutputs.shape);
		this.output.tmat = this.activation.forwardProp(h.tmat);
		// this.output.tmat = this.activation.forwardProp(addTensor(this.convOutputs, this.b).tmat);
		
		if (this.useDropout) {
			if (this.isTraining) {
				this.dropoutTensor = new Tensor(this.output.shape);
				this.dropoutTensor.tmat = this.dropout(this.dropoutTensor.tmat, this.dropoutP);
			} else {
				this.dropoutTensor = new Tensor(this.output.shape, this.dropoutP);
			}
			
			this.output.tmat.arrayTimesEquals(this.dropoutTensor.tmat);
		}
		
	}
	
	public void backProp(Tensor convoutGrad) {
		int numImgs = convoutGrad.shape.get(0);
		int imgH = convoutGrad.shape.get(2);
		int imgW = convoutGrad.shape.get(3);
		int nChannelsConvout = this.filters.shape.get(1);
		int nChannelsImgs = this.filters.shape.get(0);
		int filH = this.filters.shape.get(2);
		int filW = this.filters.shape.get(3);
		int filMidH = filH / 2;
		int filMidW = filW / 2;
		
		Tensor bpIn = convoutGrad.clone();
		if (this.useBatchNormalization) {
		    bpIn.tmat = batchNormalizationBackward(convoutGrad.tmat);
		}
		
		Tensor deriv = this.convOutputs.cloneAndZero();
		// deriv.tmat = activation.backProp(this.convOutputs.tmat);
		deriv.tmat = activation.backProp(this.output.tmat);
		
		Tensor delta = convoutGrad.cloneAndZero();
		// delta.tmat = convoutGrad.tmat.arrayTimes(deriv.tmat);
		if (this.useDropout) {
			delta.tmat = bpIn.tmat.arrayTimes(deriv.tmat).arrayTimes(this.dropoutMat);
		} else {
			delta.tmat = bpIn.tmat.arrayTimes(deriv.tmat);
		}
		
		for (int i = 0; i < numImgs; i++) {
			for (int cConvout = 0; cConvout < nChannelsConvout; cConvout++) {
				for (int y = 0; y < imgH; y++) {
					int yOffMin = intMax(-y, -filMidH);
					int yOffMax = intMin(imgH-y, filMidH+1);
					for (int x = 0; x < imgW; x++) {
						double convoutGradValue = delta.get(i, cConvout, y, x);
						int xOffMin = intMax(-x, -filMidW);
						int xOffMax = intMin(imgW-x, filMidW+1);
						for (int yOff = yOffMin; yOff < yOffMax; yOff++) {
							for (int xOff = xOffMin; xOff < xOffMax; xOff++) {
								int imgY = y + yOff;
								int imgX = x + xOff;
								int filY = filMidW + yOff;
								int filX = filMidH + xOff;
								for (int cImgs = 0; cImgs < nChannelsImgs; cImgs++) {
									double imgValue = this.bpOutput.get(i, cImgs, imgY, imgX) + this.W.get(cImgs, cConvout, filY, filX) * convoutGradValue;
									this.bpOutput.set(i, cImgs, imgY, imgX, imgValue);
									double filterValue = this.dW.get(cImgs, cConvout, filY, filX) + this.images.get(i, cImgs, imgY, imgX) * convoutGradValue;
									this.dW.set(cImgs, cConvout, filY, filX, filterValue);
								}
							}
						}
					}
				}
			}
		}
		
		this.dW.tmat = NeuralNetUtils.scalarLeftDivide(numImgs, this.dW.tmat);
		this.db.tmat = NeuralNetUtils.scalarLeftDivide(numImgs, sumTensor(convoutGrad, this.b.shape).tmat);
		
		// this.filters.tmat.plusEquals(this.dW.tmat.times(-learningRate));
		// this.W = this.filters.clone();
		
	}
	
	private Matrix dropout(Matrix inp, double dropoutP) {
		Matrix out = new Matrix(inp.getRowDimension(), inp.getColumnDimension(), 1.0);
		
		Random random = new Random(en.menghui.android.damp.utils.RandomUtilities.seed());
		
		for (int i = 0; i < out.getRowDimension(); i++) {
			for (int j = 0; j < out.getColumnDimension(); j++) {
				// out.set(i, j, MathUtils.getBinomial(1, dropoutP));
				
				if (random.nextDouble() < this.dropoutP) { // Drop!
					out.set(i, j, 0.0);
				} else {
					out.set(i, j, 1.0);
				}
			}
		}
		
		return out;
	}
	
	public void optimize() {
		if (this.optimizer instanceof AdamOptimizer) {
			AdamOptimizer optzer = (AdamOptimizer) this.optimizer;
			List<Matrix> pags = optzer.optimize(this.mW.tmat, this.vW.tmat, this.dW.tmat, this.W.tmat, this.epochCount);
			this.mW.tmat = pags.get(0);
			this.vW.tmat = pags.get(1);
			this.dW.tmat = pags.get(2);
			this.W.tmat  = pags.get(3);
			
			pags = optzer.optimize(this.mb.tmat, this.vb.tmat, this.db.tmat, this.b.tmat, this.epochCount);
			this.mb.tmat = pags.get(0);
			this.vb.tmat = pags.get(1);
			this.db.tmat = pags.get(2);
			this.b.tmat  = pags.get(3);
			
			this.epochCount++;
		} else if (this.optimizer instanceof AdaGradOptimizer) {
			AdaGradOptimizer optzer = (AdaGradOptimizer) this.optimizer;
			List<Matrix> pags = optzer.optimize(this.mW.tmat, this.dW.tmat, this.W.tmat);
			this.mW.tmat = pags.get(0);
			this.dW.tmat = pags.get(1);
			this.W.tmat  = pags.get(2);
			
			pags = optzer.optimize(this.mb.tmat, this.db.tmat, this.b.tmat);
			this.mb.tmat = pags.get(0);
			this.db.tmat = pags.get(1);
			this.b.tmat  = pags.get(2);
		} else if (this.optimizer instanceof AdaDeltaOptimizer) {
			AdaDeltaOptimizer optzer = (AdaDeltaOptimizer) this.optimizer;
			List<Matrix> pags = optzer.optimize(this.mW.tmat, this.vW.tmat, this.dW.tmat, this.W.tmat);
			this.mW.tmat = pags.get(0);
			this.vW.tmat = pags.get(1);
			this.dW.tmat = pags.get(2);
			this.W.tmat  = pags.get(3);
			
			pags = optzer.optimize(this.mb.tmat, this.vb.tmat, this.db.tmat, this.b.tmat);
			this.mb.tmat = pags.get(0);
			this.vb.tmat = pags.get(1);
			this.db.tmat = pags.get(2);
			this.b.tmat  = pags.get(3);
		}  else if (this.optimizer instanceof GDOptimizer) {
			GDOptimizer optzer = (GDOptimizer) this.optimizer;
			List<Matrix> pags = optzer.optimize(this.dW.tmat, this.W.tmat);
			this.dW.tmat = pags.get(0);
			this.W.tmat  = pags.get(1);
			
			pags = optzer.optimize(this.db.tmat, this.b.tmat);
			this.db.tmat = pags.get(0);
			this.b.tmat  = pags.get(1);
		} else if (this.optimizer instanceof SGDOptimizer) {
			SGDOptimizer optzer = (SGDOptimizer) this.optimizer;
			List<Matrix> pags = optzer.optimize(this.mW.tmat, this.dW.tmat, this.W.tmat);
			this.mW.tmat = pags.get(0);
			this.dW.tmat = pags.get(1);
			this.W.tmat  = pags.get(2);
			
			pags = optzer.optimize(this.mb.tmat, this.db.tmat, this.b.tmat);
			this.mb.tmat = pags.get(0);
			this.db.tmat = pags.get(1);
			this.b.tmat  = pags.get(2);
		} else if (this.optimizer instanceof NetsterovOptimizer) {
			NetsterovOptimizer optzer = (NetsterovOptimizer) this.optimizer;
			List<Matrix> pags = optzer.optimize(this.mW.tmat, this.dW.tmat, this.W.tmat);
			this.mW.tmat = pags.get(0);
			this.dW.tmat = pags.get(1);
			this.W.tmat  = pags.get(2);
			
			pags = optzer.optimize(this.mb.tmat, this.db.tmat, this.b.tmat);
			this.mb.tmat = pags.get(0);
			this.db.tmat = pags.get(1);
			this.b.tmat  = pags.get(2);
		} else if (this.optimizer instanceof WindowGradOptimizer) {
			WindowGradOptimizer optzer = (WindowGradOptimizer) this.optimizer;
			List<Matrix> pags = optzer.optimize(this.mW.tmat, this.dW.tmat, this.W.tmat);
			this.mW.tmat = pags.get(0);
			this.dW.tmat = pags.get(1);
			this.W.tmat  = pags.get(2);
			
			pags = optzer.optimize(this.mb.tmat, this.db.tmat, this.b.tmat);
			this.mb.tmat = pags.get(0);
			this.db.tmat = pags.get(1);
			this.b.tmat  = pags.get(2);
		}
	}
	
	public void gd() {
		if (this.useLRDecay) {
			if (decaySteps > 0) {
				this.learningRate = decayLearningRatePerStep(this.learningRate, this.learningRateDecayFactor, this.globalStep, this.decaySteps, this.staircase);
			} else {
				this.learningRate = decayLearningRate(this.learningRate, this.learningRateDecayFactor);
			}
		}
		
		// Gradient descent parameter update
		this.W.tmat.plusEquals(this.dW.tmat.times(-this.learningRate));
		this.b.tmat.plusEquals(this.db.tmat.times(-this.learningRate));
	}
	
	public Matrix batchNormalizationForward(Matrix h) {
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
		gamma = new Matrix(xhat.getRowDimension(), xhat.getColumnDimension(), 1.0);
		Matrix gammax = gamma.arrayTimes(xhat);
		beta = new Matrix(gammax.getRowDimension(), gammax.getColumnDimension(), 0.0);
		batchNormOut = gammax.plus(beta);
		
		return batchNormOut;
	}
	
	public Matrix batchNormalizationBackward(Matrix h) {
		int N = h.getRowDimension();
		int D = h.getColumnDimension();
		Matrix dbeta = NeuralNetUtils.sum(h, 0);
		Matrix dgammax = h; // not necessary, but more understandable
		Matrix dgamma = NeuralNetUtils.sum(dgammax.arrayTimes(xhat), 0);
	    Matrix dxhat = dgammax.arrayTimes(gamma);
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
	
	private int intMax(int a, int b) {
		if (a >= b) {
			return a;
		} else {
			return b;
		}
	}
	
	private int intMin(int a, int b) {
		if (a <= b) {
			return a;
		} else {
			return b;
		}
	}
	
	private Tensor addTensor(Tensor a, Tensor b) {
		Tensor res = new Tensor(a.shape);
		
		for (int i = 0; i < a.shape.get(0); i++) {
			for (int j = 0; j < a.shape.get(1); j++) {
				for (int k = 0; k < a.shape.get(2); k++) {
					for (int l = 0; l < a.shape.get(3); l++) {
						res.set(i, j, k, l, a.get(i, j, k, l) + b.get(0, j, 0, 0));
					}
				}
			}
		}
		
		return res;
	}
	
	private Tensor sumTensor(Tensor tensor, List<Integer> shape) {
		Tensor res = new Tensor(shape);
		
		for (int j = 0; j < tensor.shape.get(1); j++) {
			double sum = 0.0;
			for (int i = 0; i < tensor.shape.get(0); i++) {
				for (int k = 0; k < tensor.shape.get(2); k++) {
					for (int l = 0; l < tensor.shape.get(3); l++) {
						sum += tensor.get(i, j, k, l);
					}
				}
			}
			
			res.set(0, j, 0, 0, sum);
		}
		
		return res;
	}
	
	// Test addTensor and sumTensor methods.
	public void testPrivateMethods() {
		Tensor tensA = new Tensor(Arrays.asList(1,5,1,1));
		tensA.tmat.set(0, 0, 1.0);
		tensA.tmat.set(0, 1, 2.0);
		tensA.tmat.set(0, 2, 3.0);
		tensA.tmat.set(0, 3, 4.0);
		tensA.tmat.set(0, 4, 5.0);
		Tensor tensB = new Tensor(Arrays.asList(2,5,2,2), 1.0);
		Tensor tensC = addTensor(tensB, tensA);
		// NeuralNetUtils.printMatrix(tensC.tmat);
		
		Tensor tensD = sumTensor(tensC, tensA.shape);
		NeuralNetUtils.printMatrix(tensD.tmat);
		Tensor tensE = Tensor.sumTensorAxises(tensC, Arrays.asList(0, 2, 3));
		NeuralNetUtils.printMatrix(tensE.tmat);
	}
	
	
}