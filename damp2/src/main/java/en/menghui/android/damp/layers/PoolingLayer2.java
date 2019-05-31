package en.menghui.android.damp.layers;

import java.util.Arrays;

import en.menghui.android.damp.arrays.Tensor;

public class PoolingLayer2 extends Layer {
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
	
	public Tensor images;
	public Tensor poolout;
	public Tensor output;
	public double[][][][][] switches;
	
	public Tensor imgsGrad;
	public Tensor bpOutput;
	
	public PoolingLayer2(int numImages, int numChannels, int imgHeight, int imgWidth, int strideY, int strideX, int poolHeight, int poolWidth) {
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
		
		this.images = new Tensor(Arrays.asList(numImages, numChannels, imgHeight, imgWidth));
		this.poolout = new Tensor(Arrays.asList(numImages, numChannels, imgHeight/strideY, imgWidth/strideX));
		this.output = new Tensor(Arrays.asList(numImages, numChannels, imgHeight/strideY, imgWidth/strideX));
		this.switches = new double[numImages][numChannels][imgHeight/strideY][imgWidth/strideX][2];
	}
	
	public void forwardProp(Tensor inpt, int miniBatchSize) {
		this.images = inpt;
		this.images.shape = inpt.shape;
		
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
								double newValue = images.get(i, c, imgY, imgX);
								if (newValue > value) {
									value = newValue;
									imgYMax = imgY;
									imgXMax = imgX;
								}
							}
						}
						
						this.output.set(i, c, yOut, xOut, value);
						this.switches[i][c][yOut][xOut][0] = imgYMax;
						this.switches[i][c][yOut][xOut][1] = imgXMax;
						
					}
				}
			}
		}
	}
	
	public void backProp(Tensor pooloutGrad) {
		this.bpOutput = new Tensor(Arrays.asList(this.images.shape.get(0), this.images.shape.get(1), this.images.shape.get(2), this.images.shape.get(3)));
		
		int nImgs = pooloutGrad.shape.get(0);
		int nChannels = pooloutGrad.shape.get(1);
		int pooloutH = pooloutGrad.shape.get(2);
		int pooloutW = pooloutGrad.shape.get(3);
		
		for (int i = 0; i < nImgs; i++) {
			for (int c = 0; c < nChannels; c++) {
				for (int y = 0; y < pooloutH; y++) {
					for (int x = 0; x < pooloutW; x++) {
						double imgY = this.switches[i][c][y][x][0];
						double imgX = this.switches[i][c][y][c][1];
						this.bpOutput.set(i, c, (int)imgY, (int)imgX, pooloutGrad.get(i, c, y, x));
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
