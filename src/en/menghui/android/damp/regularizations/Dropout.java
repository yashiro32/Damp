package en.menghui.android.damp.regularizations;

import java.util.Random;

import en.menghui.android.damp.utils.MathUtils;
import en.menghui.android.damp.utils.RandomUtilities;
import Jama.Matrix;

public class Dropout {
	public double dropoutP = 0.5;
	public Matrix dropoutMat;
	
	public Dropout() {
		
	}
	
	public Dropout(double dropoutProb) {
		this.dropoutP = dropoutProb;
	}
	
	public Matrix forwardProp(Matrix a, boolean isTraining) {
		if (isTraining) {
			this.dropoutMat = new Matrix(a.getRowDimension(), a.getColumnDimension(), 1.0);
			
			for (int i = 0; i < this.dropoutMat.getRowDimension(); i++) {
				Random random = new Random(RandomUtilities.seed());
				
				for (int j = 0; j < this.dropoutMat.getColumnDimension(); j++) {
					// this.dropoutMat.set(i, j, MathUtils.getBinomial(1, this.dropoutP));
					
					if (random.nextDouble() < this.dropoutP) { // Drop!
						this.dropoutMat.set(i, j, 0.0);
					} else {
						this.dropoutMat.set(i, j, 1.0);
					}
				}
			}
		} else {
			this.dropoutMat = new Matrix(a.getRowDimension(), a.getColumnDimension(), this.dropoutP);
		}
		
		return a.arrayTimes(this.dropoutMat);
	}
	
	public Matrix backProp(Matrix a) {
		Matrix out = a.arrayTimes(this.dropoutMat);
		
		return out;
	}
}
