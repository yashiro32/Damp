package en.menghui.android.damp.utils;

import Jama.Matrix;

public class MatrixUtils {
	public static Matrix sqrt(Matrix mat) {
		Matrix a = new Matrix(mat.getRowDimension(), mat.getColumnDimension());
		
		for (int i = 0; i < mat.getRowDimension(); i++) {
			for (int j = 0; j < mat.getColumnDimension(); j++) {
				a.set(i, j, Math.sqrt(mat.get(i, j)));
			}
		}
		
		return a;
	}
	
	public static Matrix[] createMatrixArray(int n, int row, int col, double val) {
		Matrix[] matArr = new Matrix[n];
		for (int i = 0; i < matArr.length; i++) {
			Matrix mat = new Matrix(row, col, val);
			matArr[i] = mat;
		}
		
		return matArr;
	}
	
	public static Matrix[] copyMatrixArray(Matrix[] arr) {
		Matrix[] matArr = new Matrix[arr.length];
		for (int i = 0; i < matArr.length; i++) {
			matArr[i] = arr[i].copy();
		}
		
		return matArr;
	}
	
	public static Matrix minusVector(Matrix matA, Matrix matB) {
		if (matB.getColumnDimension() != matA.getColumnDimension()) {
			throw new IllegalArgumentException("Column dimension of both Matrices must be the same!");
		}
		
		if (matB.getRowDimension() != 1) {
			throw new IllegalArgumentException("Row dimension of Matrix b must be equal to one!");
		}
		
		Matrix a = new Matrix(matA.getRowDimension(), matA.getColumnDimension());
		
		for (int i = 0; i < matA.getRowDimension(); i++) {
			for (int j = 0; j < matA.getColumnDimension(); j++) {
				a.set(i, j, matA.get(i, j) - matB.get(0, j));
			}
		}
		
		return a;
	}
	
	public static Matrix timesVector(Matrix matA, Matrix matB) {
		if (matB.getColumnDimension() != matA.getColumnDimension()) {
			throw new IllegalArgumentException("Column dimension of both Matrices must be the same!");
		}
		
		if (matB.getRowDimension() != 1) {
			throw new IllegalArgumentException("Row dimension of Matrix b must be equal to one!");
		}
		
		Matrix a = new Matrix(matA.getRowDimension(), matA.getColumnDimension());
		
		for (int i = 0; i < matA.getRowDimension(); i++) {
			for (int j = 0; j < matA.getColumnDimension(); j++) {
				a.set(i, j, matA.get(i, j) * matB.get(0, j));
			}
		}
		
		return a;
	}
	
	
}
