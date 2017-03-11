package en.menghui.android.damp.utils;

import java.util.Random;

import Jama.Matrix;

public class NeuralNetUtils {
	public static Matrix sigmoid(Matrix a, boolean derivative) {
		Matrix z = new Matrix(a.getRowDimension(), a.getColumnDimension());
		
		Matrix oneMat = new Matrix(a.getRowDimension(), a.getColumnDimension(), 1.0);
		Matrix zeroMat = new Matrix(a.getRowDimension(), a.getColumnDimension(), 0.0);
		
		if (derivative) {
			z = a.arrayTimes(oneMat.minus(a));
		} else {
			// Matrix negA = zeroMat.minus(a);
			Matrix negA = a.uminus();
			
			Matrix expMat = exp(negA);
			
			z = scalarRightDivide(1.0, oneMat.plus(expMat));
		}
		
		return z;
	}
	
	public static Matrix tanh(Matrix a, boolean derivative) {
		Matrix z = new Matrix(a.getRowDimension(), a.getColumnDimension());
		
		Matrix oneMat = new Matrix(a.getRowDimension(), a.getColumnDimension(), 1.0);
		
		if (derivative) {
			Matrix e = exp(a.times(2.0));
			z = e.minus(oneMat).arrayLeftDivide(e.plus(oneMat));
		} else {
			for (int i = 0; i < a.getRowDimension(); i++) {
				for (int j = 0; j < a.getColumnDimension(); j++) {
					z.set(i, j, Math.tanh(a.get(i, j)));
				}
			}
		}
		
		return z;
	}
	
	public static Matrix relu(Matrix a, boolean derivative) {
		Matrix z = new Matrix(a.getRowDimension(), a.getColumnDimension());
		
		if (derivative) {
			for (int i = 0; i < a.getRowDimension(); i++) {
				for (int j = 0; j < a.getColumnDimension(); j++) {
					if (a.get(i,  j) > 0.0) {
						z.set(i, j, 1.0);
					}
				}
			}
		} else {
			z = maximum(0.0, a);
		}
		
		return z;
	}
	
	public static Matrix exp(Matrix inp) {
		Matrix out = new Matrix(inp.getRowDimension(), inp.getColumnDimension());
		
		for (int i = 0; i < inp.getRowDimension(); i++) {
			for (int j = 0; j < inp.getColumnDimension(); j++) {
				out.set(i, j, Math.exp(inp.get(i, j)));
			}
		}
		
		return out;
	}
	
	public static Matrix scalarRightDivide(double scalar, Matrix inp) {
		Matrix out = new Matrix(inp.getRowDimension(), inp.getColumnDimension());
		
		for (int i = 0; i < inp.getRowDimension(); i++) {
			for (int j = 0; j < inp.getColumnDimension(); j++) {
				out.set(i, j, scalar / inp.get(i, j));
			}
		}
		
		return out;
	}
	
	public static Matrix scalarLeftDivide(double scalar, Matrix inp) {
		Matrix out = new Matrix(inp.getRowDimension(), inp.getColumnDimension());
		
		for (int i = 0; i < inp.getRowDimension(); i++) {
			for (int j = 0; j < inp.getColumnDimension(); j++) {
				out.set(i, j, inp.get(i, j) / scalar);
			}
		}
		
		return out;
	}
	
	public static Matrix add(Matrix a, Matrix b) {
		/* if (b.getRowDimension() > 1 && b.getColumnDimension() > 1) {
			throw new Exception("Operation cannot be performed because of wrong matrix dimensions.");
		} */
		
		Matrix c = new Matrix(a.getRowDimension(), a.getColumnDimension());
		
		if (a.getColumnDimension() == b.getColumnDimension()) {
			for (int i = 0; i < a.getRowDimension(); i++) {
				for (int j = 0; j < b.getColumnDimension(); j++) {
					c.set(i, j, a.get(i, j) + b.get(0, j));
				}
			}
		} else if (a.getRowDimension() == b.getRowDimension()) {
			for (int j = 0; j < a.getColumnDimension(); j++) {
				for (int i = 0; i < b.getRowDimension(); i++) {
					c.set(i, j, a.get(i, j) + b.get(i, 0));
				}
			}
		}
		
		return c;
	}
	
	public static Matrix initRandomMatrix(Matrix inp) {
		Matrix out = new Matrix(inp.getRowDimension(), inp.getColumnDimension());
		
		for (int i = 0; i < inp.getRowDimension(); i++) {
			for (int j = 0; j < inp.getColumnDimension(); j++) {
				out.set(i, j, Math.random());
			}
		}
		
		return out;
	}
	
	public static Matrix initRandomMatrix(int row, int col, double lowerLimit, double upperLimit) {
		Matrix out = new Matrix(row, col);
		
		Random r = new Random();
		
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				double range = upperLimit - lowerLimit;
				double scaled = r.nextDouble() * range;
				double shifted = scaled + lowerLimit;
				
				out.set(i, j, shifted);
			}
		}
		
		return out;
	}
	
	public static Matrix combineMatrixHorizontal(Matrix a, Matrix b) {
		Matrix c = new Matrix(a.getRowDimension(), a.getColumnDimension() + b.getColumnDimension());
		
		c.setMatrix(0, a.getRowDimension()-1, 0, a.getColumnDimension()-1, a);
		c.setMatrix(0, a.getRowDimension()-1, a.getColumnDimension(), a.getColumnDimension()+b.getColumnDimension()-1, b);
		
		return c;
	}
	
	public static Matrix combineMatrixVertical(Matrix a, Matrix b) {
		Matrix c = new Matrix(a.getRowDimension() + b.getRowDimension(), a.getColumnDimension());
		
		c.setMatrix(0, a.getRowDimension()-1, 0, a.getColumnDimension()-1, a);
		c.setMatrix(a.getRowDimension(), a.getRowDimension()+b.getRowDimension()-1, 0, a.getColumnDimension()-1, b);
		
		return c;
	}
	
	public static Matrix argmax(Matrix inp) throws Exception {
		if (inp.getColumnDimension() > 1) {
			throw new Exception("Operation cannot be performed because of wrong matrix dimensions.");
		}
		
		Matrix y = new Matrix(inp.getRowDimension(), 1);
		
		y = new Matrix(inp.getRowDimension(), 1);
		for (int  i = 0; i < inp.getRowDimension(); i++) {
			double max = 0.0;
			
			if (inp.get(i, 0) >= 0.5) {
				max = 1.0;
			} else {
				max = 0.0;
			}
			
			y.set(i, 0, max);
		}
		
		return y;
	}
	
	public static Matrix argmax(Matrix inp, int axis) {
		Matrix y = new Matrix(inp.getRowDimension(), 1);
		
		if (axis == 1) {
			y = new Matrix(inp.getRowDimension(), 1);
			for (int  i = 0; i < inp.getRowDimension(); i++) {
				Matrix row = inp.getMatrix(i, i, 0, inp.getColumnDimension()-1);
				
				double index = 0.0;
				double max = 0.0;
				
				for (int j = 0; j < inp.getColumnDimension(); j++) {
					if (row.get(0, j) > max) {
						max = row.get(0, j);
						index = j + 0.0;
					}
				}
				
				y.set(i, 0, index);
			}
		}
		
		return y;
	}
	
	public static Matrix sum(Matrix inp, int axis) {
		Matrix out = null;
		
		if (axis == 0) {
			out = new Matrix(1, inp.getColumnDimension());
			
			for (int j = 0; j < inp.getColumnDimension(); j++) {
				double sum = 0.0;
				
				for (int i = 0; i < inp.getRowDimension(); i++) {
					sum += inp.get(i, j);
				}
				
				out.set(0, j, sum);
			}
		} else if (axis == 1) {
			out = new Matrix(inp.getRowDimension(), 1);
			
			for (int i = 0; i < inp.getRowDimension(); i++) {
				double sum = 0.0;
				
				for (int j = 0; j < inp.getColumnDimension(); j++) {
					sum += inp.get(i, j);
				}
				
				out.set(i, 0, sum);
			}
		} else if (axis < 0) {
			out = new Matrix(1, 1);
			double sum = 0.0;
			
			for (int i = 0; i < inp.getRowDimension(); i++) {
				for (int j = 0; j < inp.getColumnDimension(); j++) {
					sum += inp.get(i, j);
				}
			}
			
			out.set(0, 0, sum);
		}
		
		return out;
	}
	
	public static Matrix maximum(double value, Matrix inp) {
		Matrix out = inp.copy();
		
		for (int i = 0; i < inp.getRowDimension(); i++) {
			for (int j = 0; j < inp.getColumnDimension(); j++) {
				if (value > inp.get(i, j)) {
					out.set(i, j, value);
				}
			}
		}
		
		return out;
	}
	
	public static void printMatrix(Matrix mat) {
		System.out.println("No of rows: " + mat.getRowDimension());
		System.out.println("No of columns: " + mat.getColumnDimension());
		
		for (int i = 0; i < mat.getRowDimension(); i++) {
			for (int j = 0; j < mat.getColumnDimension(); j++) {
				System.out.println("Cell: " + mat.get(i, j));
			}
		}
	}
	
	public static void printMatrix(Matrix mat, boolean printCellValues) {
		System.out.println("No of rows: " + mat.getRowDimension());
		System.out.println("No of columns: " + mat.getColumnDimension());
		
		if(!printCellValues) {
			return;
		}
		
		for (int i = 0; i < mat.getRowDimension(); i++) {
			for (int j = 0; j < mat.getColumnDimension(); j++) {
				System.out.println("Cell: " + mat.get(i, j));
			}
		}
	}
	
	public static Matrix featureNormalize(Matrix mat, int axis) {
		int n = 0;
		if (axis == 0) {
			n = mat.getRowDimension();
		} else if (axis == 1) {
			n = mat.getColumnDimension();
		} else if (axis == -1) {
			n = mat.getRowDimension() * mat.getColumnDimension();
		}
		
		Matrix sum = new Matrix(1, 1);
		if (axis == 0) {
			sum = sum(mat, 0);
		} else if (axis == 1) {
			sum = sum(mat, 1);
		} else if (axis == -1) {
			sum = sum(mat, -1);
		}
		
		Matrix meanMat = new Matrix(mat.getRowDimension(), mat.getColumnDimension());
		Matrix nMat = new Matrix(1, 1);
		if (axis == 0) {
			nMat = new Matrix(1, mat.getColumnDimension(), 1.0/n);
			
			Matrix mMat = sum.arrayTimes(nMat);
			for (int i = 0; i < n; i++) {
				meanMat.setMatrix(i, i, 0, mat.getColumnDimension()-1, mMat);
			}
		} else if (axis == 1) {
			nMat = new Matrix(mat.getRowDimension(), 1, 1.0/n);
			
			Matrix mMat = sum.arrayTimes(nMat);
			for (int i = 0; i < n; i++) {
				meanMat.setMatrix(0, mat.getRowDimension()-1, i, i, mMat);
			}
		} else if (axis == -1) {
			meanMat = new Matrix(mat.getRowDimension(), mat.getColumnDimension(), sum.get(0,0) / (mat.getRowDimension() * mat.getColumnDimension()));
		}
		
		Matrix xNormMat = mat.minus(meanMat);
		
		Matrix xNormSqrMat = xNormMat.arrayTimes(xNormMat);
		
		if (axis == 0) {
			sum = sum(xNormSqrMat, 0);
		} else if (axis == 1) {
			sum = sum(xNormSqrMat, 1);
		} else if (axis == -1) {
			sum = sum(xNormSqrMat, -1);
		}
		
		Matrix stdMat = new Matrix(1, 1);
		if (axis == 0 || axis == 1) {
			Matrix varianceMat = sum.arrayTimes(nMat);
			
			Matrix stMat = MatrixUtils.sqrt(varianceMat);
			stdMat = new Matrix(mat.getRowDimension(), mat.getColumnDimension());
			
			if (axis == 0) {
				for (int i = 0; i < n; i++) {
					stdMat.setMatrix(i, i, 0, mat.getColumnDimension()-1, stMat);
			    }
			} else if (axis == 1) {
				for (int i = 0; i < n; i++) {
					stdMat.setMatrix(0, mat.getRowDimension()-1, i, i, stMat);
			    }
			}
		} else if (axis == -1) {
			Matrix varianceMat = new Matrix(mat.getRowDimension(), mat.getColumnDimension(), sum.get(0,0) / (mat.getRowDimension() * mat.getColumnDimension()));
			stdMat = MatrixUtils.sqrt(varianceMat);
		}
		
		Matrix normMat = xNormMat.arrayRightDivide(stdMat);
		
		return normMat;
	}
	
	public static Matrix featureNormalizeAxisZero(Matrix mat) {
		int n = mat.getRowDimension();
		Matrix sum = sum(mat, 0);
		// Matrix sum = sum(mat, -1);
		Matrix nMat = new Matrix(1, mat.getColumnDimension(), 1.0/n);
		// Matrix meanMat = new Matrix(mat.getRowDimension(), mat.getColumnDimension(), sum.get(0,0) / (mat.getRowDimension() * mat.getColumnDimension()));
		Matrix mMat = sum.arrayTimes(nMat);
		Matrix meanMat = new Matrix(mat.getRowDimension(), mat.getColumnDimension());
		for (int i = 0; i < n; i++) {
			// meanMat.setMatrix(0, mat.getRowDimension()-1, i, i, mMat);
			meanMat.setMatrix(i, i, 0, mat.getColumnDimension()-1, mMat);
		}
		
		Matrix xNormMat = mat.minus(meanMat);
		
		Matrix xNormSqrMat = xNormMat.arrayTimes(xNormMat);
		sum = sum(xNormSqrMat, 0);
		// sum = sum(xNormSqrMat, -1);
		Matrix varianceMat = sum.arrayTimes(nMat);
		// Matrix varianceMat = new Matrix(mat.getRowDimension(), mat.getColumnDimension(), sum.get(0,0) / (mat.getRowDimension() * mat.getColumnDimension()));;
		// Matrix stdMat = MatrixUtils.sqrt(varianceMat);
		Matrix stMat = MatrixUtils.sqrt(varianceMat);
		Matrix stdMat = new Matrix(mat.getRowDimension(), mat.getColumnDimension());
		for (int i = 0; i < n; i++) {
			// stdMat.setMatrix(0, mat.getRowDimension()-1, i, i, stMat);
			stdMat.setMatrix(i, i, 0, mat.getColumnDimension()-1, stMat);
		}
		
		Matrix normMat = xNormMat.arrayRightDivide(stdMat);
		
		return normMat;
	}
	
	public static Matrix featureNormalizeAxisOne(Matrix mat) {
		int n = mat.getColumnDimension();
		Matrix sum = sum(mat, 1);
		Matrix nMat = new Matrix(mat.getRowDimension(), 1, 1.0/n);
		
		Matrix mMat = sum.arrayTimes(nMat);
		Matrix meanMat = new Matrix(mat.getRowDimension(), mat.getColumnDimension());
		for (int i = 0; i < n; i++) {
			meanMat.setMatrix(0, mat.getRowDimension()-1, i, i, mMat);
		}
		
		Matrix xNormMat = mat.minus(meanMat);
		
		Matrix xNormSqrMat = xNormMat.arrayTimes(xNormMat);
		sum = sum(xNormSqrMat, 1);
		Matrix varianceMat = sum.arrayTimes(nMat);
		Matrix stMat = MatrixUtils.sqrt(varianceMat);
		Matrix stdMat = new Matrix(mat.getRowDimension(), mat.getColumnDimension());
		for (int i = 0; i < n; i++) {
			stdMat.setMatrix(0, mat.getRowDimension()-1, i, i, stMat);
		}
		
		Matrix normMat = xNormMat.arrayRightDivide(stdMat);
		
		return normMat;
	}
	
	public static Matrix featureNormalizeSingle(Matrix mat) {
		int n = mat.getRowDimension();
		
		Matrix sum = sum(mat, -1);
		
		Matrix meanMat = new Matrix(mat.getRowDimension(), mat.getColumnDimension(), sum.get(0,0) / (mat.getRowDimension() * mat.getColumnDimension()));
		
		Matrix xNormMat = mat.minus(meanMat);
		
		Matrix xNormSqrMat = xNormMat.arrayTimes(xNormMat);
		
		sum = sum(xNormSqrMat, -1);
		
		Matrix varianceMat = new Matrix(mat.getRowDimension(), mat.getColumnDimension(), sum.get(0,0) / (mat.getRowDimension() * mat.getColumnDimension()));
		Matrix stdMat = MatrixUtils.sqrt(varianceMat);
		
		Matrix normMat = xNormMat.arrayRightDivide(stdMat);
		
		return normMat;
	}  
	
} 
