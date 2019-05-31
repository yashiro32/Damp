package en.menghui.android.damp.utils;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Volume {
	public int depth;
	public int height;
	public double[] weightGradients;
	public double[] weights;
	public int width;
	public int length;
	
	/**
	 * Volume will be filled with random numbers.
	 * 
	 * @param width
	 * @param height
	 * @param depth
	 */
	public Volume(int width, int height, int depth) {
		this.width = width;
		this.height = height;
		this.depth = depth;
		
		int n = width * height * depth;
		this.weights = new double[n];
		this.weightGradients = new double[n];
		
		// weight normalization is done to equalize the output
		// variance of every neuron, otherwise neurons with a lot
		// of incoming connections have outputs of larger variance.
		double scale = Math.sqrt(1.0 / (width * height * depth));
		
		for (int i = 0; i < n; i++) {
			this.weights[i] = RandomUtilities.randn(0.0, scale);
		}
	}
	
	/**
	 * @param width
	 * @param height
	 * @param depth
	 * @param c value to initialize the volume with.
	 */
	public Volume(int width, int height, int depth, double c) {
		// We were given dimensions of the vol.
		this.width = width;
		this.height = height;
		this.depth = depth;
		
		int n = width * height * depth;
		this.weights = new double[n];
		this.weightGradients = new double[n];
		
		if (c != 0) {
			for (int i = 0; i < n; i++) {
				this.weights[i] = c;
			}
		}
	}
	
	public Volume(List<Double> weights) {
		// We were given a list in weights, assume 1D volume and fill it up.
		this.width = 1;
		this.height = 1;
		this.depth = weights.size();
		
		this.weights = new double[this.depth];
		this.weightGradients = new double[this.depth];
		
		for (int i = 0; i < this.depth; i++) {
			this.weights[i] = weights.get(i);
		}
	}
	
	public double get(int x, int y, int d) {
		int ix = ((this.width * y) + x) * this.depth + d;
		
		return this.weights[ix];
	}
	
	public void set(int x, int y, int d, double v) {
		int ix = ((this.width * y) + x) * this.depth + d;
		
		this.weights[ix] += v;
	}
	
	public void add(int x, int y, int d, double v) {
		int ix = ((this.width * y) + x) * this.depth + d;
		
		this.weights[ix] += v;
	}
	
	public double getGradient(int x, int y, int d) {
		int ix = ((this.width * y) + x) * this.depth + d;
		
		return this.weightGradients[ix];
	}
	
	public void setGradient(int x, int y, int d, double v) {
		int ix = ((this.width * y) + x) * this.depth + d;
		
		this.weightGradients[ix] = v;
	}
	
	public void addGradient(int x, int y, int d, double v) {
		int ix = ((this.width * y) + x) * this.depth + d;
		
		this.weightGradients[ix] += v;
	}
	
	public Volume cloneAndZero() {
		return new Volume(this.width, this.height, this.depth, 0.0);
	}
	
	public Volume Clone() {
		Volume volume = new Volume(this.width, this.height, this.depth, 0.0);
		int n = this.weights.length;
		
		for (int i = 0; i < n; i++) {
			volume.weights[i] = this.weights[i];
		}
		
		return volume;
	}
	
	public void zeroGradients() {
		Arrays.fill(this.weightGradients, 0.0);
	}
	
	public void addFrom(Volume volume) {
		for (int i = 0; i < this.weights.length; i++) {
			this.weights[i] += volume.weights[i];
		}
	}
	
	public void addGradientFrom(Volume volume) {
		for (int i = 0; i < this.weightGradients.length; i++) {
			// this.weightGradients[i] += volume.weightGradients[i];
			
			this.weightGradients[i] += volume.getGradient(i);
		}
	}
	
	public void addFromScaled(Volume volume, double a) {
		for (int i = 0; i < this.weights.length; i++) {
			// this.weights[i] += a * volume.weights[i];
			
			 this.weights[i] += a * volume.get(i);
		}
	}
	
	public void setConst(double c) {
		for (int i = 0; i < this.weights.length; i++) {
			this.weights[i] += c;
		}
	}
	
	public double get(int i) {
		return this.weights[i];
	}
	
	public double getGradient(int i) {
		return this.weightGradients[i];
	}
	
	public void setGradient(int i, double v) {
		this.weightGradients[i] = v;
	}
	
	public void set(int i, double v) {
		this.weights[i] = v;
	}
	
	public int getLength() {
		return this.weights.length;
	}
	
	
}
