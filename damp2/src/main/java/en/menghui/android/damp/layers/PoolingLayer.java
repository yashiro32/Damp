package en.menghui.android.damp.layers;

import Jama.Matrix;

public class PoolingLayer extends Layer {
	public int numImages;
	public int numChannels;
	public int imageWidth;
	public int imageHeight;
	public int strideX;
	public int strideY;
	
	public int poolHeight;
	public int poolWidth;
	
	public int outHeight;
	public int outWidth;
	
	public int poolHTop;
	public int poolHBottom;
	public int poolWLeft;
	public int poolWRight;
	
	public Matrix[][] images;
	public Matrix[][] poolout;
	public double[][][][][] switches;
	
	public Matrix[][] imgsGrad;
	
	public PoolingLayer(int numImages, int numChannels, int imgHeight, int imgWidth, int strideY, int strideX, int poolHeight, int poolWidth) {
		this.type = "pooling";
		this.numImages = numImages;
		this.numChannels = numChannels;
		this.imageWidth = imgWidth;
		this.imageHeight = imgHeight;
		this.strideX = strideX;
		this.strideY = strideY;
		
		this.poolHeight = poolHeight;
		this.poolWidth = poolWidth;
		
		this.outHeight = imgHeight / strideY;
		this.outWidth = imgWidth / strideX;
		
		this.poolHTop = poolHeight / 2 - 1 + poolHeight % 2;
		this.poolHBottom = poolHeight / 2 + 1;
		this.poolWLeft = poolWidth / 2 - 1 + poolWidth % 2;
		this.poolWRight = poolWidth / 2 + 1;
		
		this.images = new Matrix[numImages][numChannels];
		this.poolout = new Matrix[numImages][numChannels];
		this.switches = new double[numImages][numChannels][imageHeight/strideY][imageWidth/strideX][2];
	}
	
	public void forwardProp(Matrix[][] inpt, int miniBatchSize) {
		this.images= inpt;
		
		double imgYMax = 0.0;
		double imgXMax = 0.0;
		
		for (int i = 0; i < numImages; i++) {
			for (int c = 0; c < numChannels; c++) {
				for (int yOut = 0; yOut < outHeight; yOut++) {
					int y = yOut * strideY;
					int yMin = intMax(y-poolHTop, 0);
					int yMax = intMin(y+poolHBottom, imageHeight);
					for (int xOut = 0; xOut < outWidth; xOut++) {
						int x = xOut * strideX;
						int xMin = intMax(x-poolWLeft, 0);
						int xMax = intMin(x+poolWRight, imageWidth);
						double value = Double.NEGATIVE_INFINITY;
						for (int imgY = yMin; imgY < yMax; imgY++) {
							for (int imgX = xMin; imgX < xMax; imgX++) {
								double newValue = images[i][c].get(imgY, imgX);
								if (newValue > value) {
									value = newValue;
									imgYMax = imgY;
									imgXMax = imgX;
								}
							}
						}
						
						poolout[i][c].set(yOut, xOut, value);
						switches[i][c][yOut][xOut][0] = imgYMax;
						switches[i][c][yOut][xOut][1] = imgXMax;
						
					}
				}
			}
		}
	}
	
	public void backProp(Matrix[][] pooloutGrad) {
		imgsGrad = new Matrix[this.images.length][this.images[0].length];
		
		for (int i = 0; i < imgsGrad.length; i++) {
			for (int j = 0; j < imgsGrad[0].length; j++) {
				Matrix mat = new Matrix(this.images[0][0].getRowDimension(), this.images[0][0].getColumnDimension());
				imgsGrad[i][j] = mat;
			}
		}
		
		int nImgs = pooloutGrad.length;
		int nChannels = pooloutGrad[0].length;
		int pooloutH = pooloutGrad[0][0].getRowDimension();
		int pooloutW = pooloutGrad[0][0].getColumnDimension();
		
		for (int i = 0; i < nImgs; i++) {
			for (int c = 0; c < nChannels; c++) {
				for (int y = 0; y < pooloutH; y++) {
					for (int x = 0; x < pooloutW; x++) {
						double imgY = switches[i][c][y][x][0];
						double imgX = switches[i][c][y][c][1];
						imgsGrad[i][c].set((int)imgY, (int)imgX, pooloutGrad[i][c].get(y, x));
					}
				}
			}
		}
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
	
	
}
