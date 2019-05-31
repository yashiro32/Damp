package en.menghui.android.damp.layers;

import java.util.ArrayList;
import java.util.List;

import en.menghui.android.damp.activations.Activation;
import en.menghui.android.damp.optimizations.AdaDeltaOptimizer;
import en.menghui.android.damp.optimizations.AdaGradOptimizer;
import en.menghui.android.damp.optimizations.AdamOptimizer;
import en.menghui.android.damp.optimizations.GDOptimizer;
import en.menghui.android.damp.optimizations.NetsterovOptimizer;
import en.menghui.android.damp.optimizations.Optimizer;
import en.menghui.android.damp.optimizations.SGDOptimizer;
import en.menghui.android.damp.optimizations.WindowGradOptimizer;
import en.menghui.android.damp.utils.Volume;

public class ConvLayer {
	public int width;
	public int height;
	
	public Volume biases;
	public Volume mbiases;
	public Volume vbiases;
	public List<Volume> filters;
	public List<Volume> mfilters;
	public List<Volume> vfilters;
	public int filterCount;
	
	public double l1DecayMul;
	public double l2DecayMul;
	
	public int stride;
	public int pad;
	
	public double biasPref;
	public Activation activation;
	
	public List<Volume> inputActivation;
	public List<Volume> outputActivation;
	
	public int groupSize;
	
	
	public int inputWidth;
	public int inputHeight;
	public int inputDepth;
	
	public int outputWidth;
	public int outputHeight;
	public int outputDepth;
	
	public Optimizer optimizer = new Optimizer();
	
	public int epochCount = 0; 
	
	public ConvLayer(int width, int height, int filterCount) {
		this.groupSize = 2;
		this.l1DecayMul = 0.0;
		this.l2DecayMul = 1.0;
		this.stride = 1;
		this.pad = 0;
		
		this.filterCount = filterCount;
		this.width = width;
		this.height = height;
	}
	
	public List<Volume> forwardProp(List<Volume> input) {
		this.inputActivation = input;
		
		List<Volume> outputActivation = new ArrayList<Volume>();
		for (int i = 0; i < input.size(); i++) {
			Volume volume = new Volume(this.outputWidth, this.outputHeight, this.outputDepth, 0.0);
			outputActivation.add(volume);
		}
		
		int volumeWidth = input.get(0).width;
		int volumeHeight = input.get(0).height;
		int xyStride = this.stride;
		
		for (int m = 0; m < input.size(); m++) {
			for (int depth = 0; depth < this.outputDepth; depth++) {
				Volume filter = this.filters.get(depth);
				
				int y = -this.pad;
				
				for (int ay = 0; ay < this.outputHeight; y+= xyStride, ay++) {
					// xyStride
					int x = -this.pad;
					for (int ax = 0; ax < this.outputWidth; x += xyStride, ax++) {
						// xyStride
						
						// Convolve centered at this particular location.
						double a = 0.0;
						for (int fy = 0; fy < filter.height; fy++) {
							int oy = y + fy; // Coordinates in the original input array coordinates.
							for (int fx = 0; fx < filter.width; fx++) {
								int ox = x + fx;
								if (oy >= 0 && oy < volumeHeight && ox >= 0 && ox < volumeWidth) {
									for (int fd = 0; fd < filter.depth; fd++) {
										// Avoid function call overhead (x2) for efficiency, compromise modularity.
										a += filter.weights[(filter.width * fy + fx) * filter.depth + fd] * input.get(m).weights[(volumeWidth * oy + ox) * input.get(m).depth + fd];
									}
								}
							}
						}
						
						a += this.biases.weights[depth];
						outputActivation.get(m).set(ax ,ay, depth, a);
					}
				}
			}
		}
		
		this.outputActivation = outputActivation;
		
		return this.outputActivation;
	}
	
	public void backProp() {
		List<Volume> inputs = this.inputActivation;
		
		for (int m = 0; m < inputs.size(); m++) {
			Volume volume = inputs.get(m);
			
			volume.weightGradients = new double[volume.weights.length]; // Zero out gradient wrt bottom data, we're about to fill it.
			
			int volumeWidth = volume.width;
			int volumeHeight = volume.height;
			int volumeDepth = volume.depth;
			int xyStride = this.stride;
			
			Volume temp = volume;
			for (int depth = 0; depth < this.outputDepth; depth++) {
				Volume filter = this.filters.get(depth);
				int y = -this.pad;
				for (int ay = 0; ay < this.outputHeight; y += xyStride, ay++) {
					// xyStride
					int x = -this.pad;
					for (int ax = 0; ax < this.outputWidth; x += xyStride, ax++) {
						// xyStride
						
						// Convolve centered at this particular location.
						double chainGradient = this.outputActivation.get(m).getGradient(ax, ay, depth);
						// Gradient from above, from chain rule.
						for (int fy = 0; fy < filter.height; fy++) {
							int oy = y + fy; // Coordinates in the original input array coordinates.
							for (int fx = 0; fx < filter.width; fx++) {
								int ox = x + fx;
								if (oy >= 0 && oy < volumeHeight && ox >= 0 && ox < volumeWidth) {
									for (int fd = 0; fd < filter.depth; fd++) {
										filter.addGradient(fx, fy, fd, volume.get(ox, oy, fd) * chainGradient);
										temp.addGradient(ox, oy, fd, filter.get(fx, fy, fd) * chainGradient);
									}
								}
							}
						}
						
						this.biases.weightGradients[depth] += chainGradient;
					}
				}
			
			}
		}
		
	}
	
	public void optimize() {
		if (this.optimizer instanceof AdamOptimizer) {
			AdamOptimizer optzer = (AdamOptimizer) this.optimizer;
			List<List<Volume>> pags = optzer.optimize(this.mfilters, this.vfilters, this.filters, this.epochCount);
			this.mfilters = pags.get(0);
			this.vfilters = pags.get(1);
			this.filters = pags.get(2);
			
			List<Volume> pag = optzer.optimize(this.mbiases, this.vbiases, this.biases, this.epochCount);
			this.mbiases = pag.get(0);
			this.vbiases = pag.get(1);
			this.biases = pag.get(2);
			
			this.epochCount++;
		}
	}
	
	public void init(int inputWidth, int inputHeight, int inputDepth) {
		// super.init(inputWidth, inputHeight, inputDepth);
		
		this.inputWidth = inputWidth;
		this.inputHeight = inputHeight;
		this.inputDepth = inputDepth;
		
		// Required.
		this.outputDepth = this.filterCount;
		
		// Computed
		// note we are doing floor, so if the strided convolution of the filter doesn't fit into the input
		// volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
		// final application.
		this.outputWidth = (int)Math.floor((this.inputWidth + this.pad * 2 - this.width) / (double)this.stride + 1);
		this.outputHeight = (int)Math.floor((this.inputHeight + this.pad * 2 - this.height) / (double)this.stride + 1);
		
		// Initializations.
		double bias = this.biasPref;
		this.filters = new ArrayList<Volume>();
		this.mfilters = new ArrayList<Volume>();
		this.vfilters = new ArrayList<Volume>();
		
		for (int i = 0; i < this.outputDepth; i++) {
			this.filters.add(new Volume(this.width, this.height, this.inputDepth));
			this.mfilters.add(new Volume(this.width, this.height, this.inputDepth));
			this.vfilters.add(new Volume(this.width, this.height, this.inputDepth));
		}
		
		this.biases = new Volume(1, 1, this.outputDepth, bias);
		this.mbiases = new Volume(1, 1, this.outputDepth, bias);
		this.vbiases = new Volume(1, 1, this.outputDepth, bias);
	}
	
	
}
