package en.menghui.android.damp;

import Jama.Matrix;

public class ConvolutionLayer {
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
	
	public Matrix[][] images;
	public Matrix[][] filters;
	public Matrix[][] convOutputs;
	
	public Matrix[][] W;
	public Matrix b;
	
	public double regLambda = 0.01;
	public double learningRate = 0.01;
	
	public ConvolutionLayer(int numImages, int numFilters, int numChannelsIn, int numChannelsOut, int imgWidth, int imgHeight, int filterWidth, int filterHeight) {
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
		
		images = new Matrix[numImages][numChannelsIn];
		filters = new Matrix[numChannelsIn][numChannelsOut];
		convOutputs = new Matrix[numImages][numChannelsOut];
		
		this.W = new Matrix[numImages][numChannelsIn];
		this.b = new Matrix(1, numChannelsIn, 0.0);
		
		// Initialize the filters with random values.
		for (int i = 0; i < numChannelsIn; i++) {
			for (int j = 0; j < numChannelsOut; j++) {
				Matrix mat = Matrix.random(filterHeight, filterWidth);
				filters[i][j] = mat;
			}
		}
		
		// Initialize the convolution outputs with random values.
		for (int i = 0; i < numChannelsIn; i++) {
			for (int j = 0; j < numChannelsOut; j++) {
				Matrix mat = Matrix.random(imgHeight, imgWidth);
				convOutputs[i][j] = mat;
			}
		}
	}
	
	public void setInput(Matrix[][] inpt, int miniBatchSize) {
		this.images = inpt;
		
		for (int i = 0; i < numImages; i++) {
			for (int cOut = 0; cOut < numChannelsOut; cOut++) {
				for (int y = 0; y < imageHeight; y++) {
					int yOffMin = intMax(-y, -filMidH);
					int yOffMax	= intMin(imageWidth-y, filMidW+1);
					for (int x = 0; x < imageHeight; x++) {
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
									value += images[i][cIn].get(imageY, imageX) * filters[cIn][cOut].get(filY, filX);
								}
							}
						}
						convOutputs[i][cOut].set(y, x, value);
					}
				}
			}
		}
	}
	
	public void backProp(Matrix[][] imgs, Matrix[][] convoutGrad, Matrix[][] filters, Matrix[][] imgsGrad, Matrix[][] filtersGrad) {
		int numImgs = convoutGrad.length;
		int imgH = convoutGrad[0][0].getRowDimension();
		int imgW = convoutGrad[0][0].getColumnDimension();
		int nChannelsConvout = filters[0].length;
		int nChannelsImgs = filters.length;
		int filH = filters[0][0].getRowDimension();
		int filW = filters[0][0].getColumnDimension();
		int filMidH = filH / 2;
		int filMidW = filW / 2;
		
		for (int i = 0; i < numImgs; i++) {
			for (int cConvout = 0; cConvout < nChannelsConvout; cConvout++) {
				for (int y = 0; y < imgH; y++) {
					int yOffMin = intMax(-y, -filMidH);
					int yOffMax = intMin(imgH-y, filMidH+1);
					for (int x = 0; x < imgW; x++) {
						double convoutGradValue = convoutGrad[i][cConvout].get(y, x);
						int xOffMin = intMax(-x, -filMidW);
						int xOffMax = intMin(imgW-x, filMidW+1);
						for (int yOff = yOffMin; yOff < yOffMax; yOff++) {
							for (int xOff = xOffMin; xOff < xOffMax; xOff++) {
								int imgY = y + yOff;
								int imgX = x + xOff;
								int filY = filMidW + yOff;
								int filX = filMidH + xOff;
								for (int cImgs = 0; cImgs < nChannelsImgs; cImgs++) {
									double imgValue = imgsGrad[i][cImgs].get(imgY, imgX) + filters[cImgs][cConvout].get(filY, filX) * convoutGradValue;
									imgsGrad[i][cImgs].set(imgY, imgX, imgValue);
									double filterValue = filtersGrad[cImgs][cConvout].get(filY, filX) + imgs[i][cImgs].get(imgY, imgX) * convoutGradValue;
									filtersGrad[cImgs][cConvout].set(filY, filX, filterValue);
								}
							}
						}
					}
				}
			}
		}
		
		for (int i = 0; i < filtersGrad.length; i++) {
			for (int j = 0; j < filtersGrad[0].length; j++) {
				NeuralNetUtils.scalarLeftDivide(numImgs, filtersGrad[i][j]);
			}
		}
		
		int nImgs = convoutGrad.length;
		
		for (int i = 0; i < filtersGrad.length; i++) {
			for (int j = 0; j < filtersGrad[0].length; j++) {
				this.W[i][j].plusEquals(filtersGrad[i][j].times(-learningRate));
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
