package en.menghui.android.damp.arrays;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import android.annotation.SuppressLint;
import android.util.Log;
import Jama.Matrix;

public class Tensor {
	private static final String TAG = "Tensor";
	public Matrix tmat;
	public List<Integer> shape = new ArrayList<Integer>();
	
	public Tensor(List<Integer> shape) {
		this.shape = shape;
		
		tmat = new Matrix(shape.get(0), getColumns());
	}
	
	public Tensor(List<Integer> shape, double value) {
		this.shape = shape;
		
		tmat = new Matrix(shape.get(0), getColumns(), value);
	}
	
	public Tensor(List<Integer> shape, boolean random) {
		this.shape = shape;
		
		if (random) {
			this.tmat = Matrix.random(shape.get(0), getColumns());
		} else { 
			tmat = new Matrix(shape.get(0), getColumns());
		}
	}
	
	private int getSize() {
		int size = 1;
		
		for (int i = 0; i < this.shape.size(); i++) {
			size *= this.shape.get(i);
		}
		
		return size;
	}
	
	private int getColumns() {
		int columns = 1;
		if (this.shape.size() > 1) {
			for (int i = 1; i < this.shape.size(); i++) {
				columns *= this.shape.get(i);
		    }
		}
		
		return columns;
	}
	
	public double get(int n, int d, int y, int x) {
		int ix = ((this.shape.get(2) * d) + y) * this.shape.get(3) + x;
		
		// Log.i(TAG, "IX: " + ix);
		
		return this.tmat.get(n, ix);
	}
	
	public double get(List<Integer> indexes) {
		int ix = 0;
		
		if (indexes.size() > 1) {
			ix = indexes.get(1);
			
			if (indexes.size() > 2) {
				// ix = (this.shape.get(2) * ix) + indexes.get(2);
			    for (int i = 2; i < indexes.size(); i++) {
			    	ix *= this.shape.get(i);
			    	ix += indexes.get(i);
			    }
			}
		
		}
		
		// Log.i(TAG, "IX: " + ix);
		
		return this.tmat.get(indexes.get(0), ix);
	}
	
	public void set(int n, int d, int y, int x, double v) {
		int ix = ((this.shape.get(2) * d) + y) * this.shape.get(3) + x;
		
		this.tmat.set(n, ix, v);
	}
	
	public void set(List<Integer> indexes, double v) {
		int ix = 0;
		
		if (indexes.size() > 1) {
			ix = indexes.get(1);
			
			if (indexes.size() > 2) {
			    for (int i = 2; i < indexes.size(); i++) {
			    	ix *= this.shape.get(i);
			    	ix += indexes.get(i);
			    }
			}
		
		}
		
		this.tmat.set(indexes.get(0), ix, v);
	}
	
	public void add(int n, int d, int y, int x, double v) {
		int ix = ((this.shape.get(2) * d) + y) * this.shape.get(3) + x;
		
		this.tmat.set(n, ix, this.tmat.get(n, ix) + v);
	}
	
	public Tensor cloneAndZero() {
		return new Tensor(this.shape, 0.0);
	}
	
	public Tensor clone() {
		Tensor tensor = new Tensor(this.shape, 0.0);
		int r = this.tmat.getRowDimension();
		int c = this.tmat.getColumnDimension();
		
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				tensor.tmat.set(i, j, this.tmat.get(i, j));
			}
		}
		
		return tensor;
	}
	
	public void zeroGradients() {
		Arrays.fill(this.tmat.getArray(), 0.0);
	}
	
	public void addFrom(Tensor tensor) {
		this.tmat.plusEquals(tensor.tmat);
	}
	
	public void addFromScaled(Tensor tensor, double a) {
		this.tmat.plusEquals(tensor.tmat.times(a));
	}
	
	public void setConst(double c) {
		for (int i = 0; i < this.tmat.getRowDimension(); i++) {
			for (int j = 0; j < this.tmat.getColumnDimension(); j++) {
				this.tmat.set(i, j, this.tmat.get(i, j) + c);
			}
		}
	}
	
	@SuppressLint("UseSparseArrays")
	public static Tensor sumTensorAxises(Tensor tensor, List<Integer> axis) {
		List<Integer> reshape = new ArrayList<Integer>();
		List<Integer> shape = new ArrayList<Integer>();
		
		Map<Integer, Integer> reshapeMap = new HashMap<Integer, Integer>();
		Map<Integer, Integer> shapeMap = new HashMap<Integer, Integer>();
		
		int reshapeCount = 0;
		int shapeCount = 0;
		for (int i = 0; i < tensor.shape.size(); i++) {
			if (axis.indexOf(i) == -1) {
				reshape.add(tensor.shape.get(i));
				reshapeMap.put(i, reshapeCount);
				reshapeCount++;
			} else {
				shape.add(tensor.shape.get(i));
				shapeMap.put(i, shapeCount);
				shapeCount++;
			}
		}
		
		Tensor res = new Tensor(reshape);
		
		int reshapeSum = 1;
		for (int i = 0; i < reshape.size(); i++) {
			reshapeSum *= reshape.get(i);
		}
		
		int shapeSum = 1;
		for (int i = 0; i < shape.size(); i++) {
			shapeSum *= shape.get(i);
		}
		
		List<Integer> reshape2 = new ArrayList<Integer>();
		List<Integer> shape2 = new ArrayList<Integer>();
		for (int a = 0; a < reshapeSum; a++) {
			reshape2 = new ArrayList<Integer>();
			int reshapeSize = 1;
			for (int b = 0; b < reshape.size(); b++) {
				reshapeSize *= reshape.get(b);
				
				reshape2.add((a / (reshapeSize / reshape.get(b))) % reshape.get(b));
			}
			
			Collections.reverse(reshape2);
			// Log.i(TAG, "Reshape 2: " + reshape2.toString());
			
			
			double sum = 0.0;
			for (int c = 0; c < shapeSum; c++) {
				shape2 = new ArrayList<Integer>();
				int shapeSize = 1;
			    for (int d = 0; d < shape.size(); d++) {
			    	shapeSize *= shape.get(d);
			    	
					shape2.add((c / (shapeSize / shape.get(d))) % shape.get(d));
			    }
			    
			    Collections.reverse(shape2);
			    // Log.i(TAG, "Shape 2: " + shape2.toString());
			    
			    List<Integer> list = new ArrayList<Integer>();
			    for (int e = 0; e < tensor.shape.size(); e++) {
			    	if (reshapeMap.containsKey(e)) {
			    		list.add(reshape2.get(reshapeMap.get(e)));
			    	} else {
			    		list.add(shape2.get(shapeMap.get(e)));
			    	}
			    }
			    
			    // Log.i(TAG, "Shape List: " + list.toString());
			    sum += tensor.get(list);
				
		    }
			
			res.set(reshape2, sum);
		}
		
		return res;
	}
	
	
}
