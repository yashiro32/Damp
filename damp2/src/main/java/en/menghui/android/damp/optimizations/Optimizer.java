package en.menghui.android.damp.optimizations;

import java.util.ArrayList;
import java.util.List;

import Jama.Matrix;

public class Optimizer {
	public String name = "";
	public String type = "";
	
	public Matrix W;
	public Matrix b;
	
	public double regLambda = 0.01;
	public double learningRate = 0.01;
	
	public double learningRateDecayFactor = 1.0;
	public boolean useLRDecay = false;
	public boolean staircase = false;
	
	public int globalStep = 0;
	public int decaySteps = 0;
	
	public Optimizer() {
		
	}
	
	public List<Matrix> optimize(Matrix m, Matrix v, Matrix d, Matrix p) {
		List<Matrix> list = new ArrayList<>();
		
		list.add(m);
		list.add(v);
		list.add(d);
		list.add(p);
		
		return list;
	}
	
	public double decayLearningRatePerStep(double lr, double decayFactor, int globalStep, int decaySteps, boolean staircase) {
		double decayedLr = 0.0;
		
		if (staircase) {
			decayedLr = lr * (Math.pow(decayFactor, (int)Math.floor(globalStep/decaySteps)));
		} else {
			decayedLr = lr * (Math.pow(decayFactor, (globalStep/decaySteps)));
		}
		
		return decayedLr;
	}
	
	public double decayLearningRate(double lr, double decayFactor) {
		return lr * decayFactor;
	}
	
	public void adjustLearningRate() {
		if (this.useLRDecay) {
			if (decaySteps > 0) {
				this.learningRate = decayLearningRatePerStep(this.learningRate, this.learningRateDecayFactor, this.globalStep, this.decaySteps, this.staircase);
			} else {
				this.learningRate = decayLearningRate(this.learningRate, this.learningRateDecayFactor);
			}
		}
	}
}
