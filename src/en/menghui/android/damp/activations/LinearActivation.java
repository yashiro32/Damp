package en.menghui.android.damp.activations;

import Jama.Matrix;

public class LinearActivation extends Activation {
	@Override
	public Matrix forwardProp(Matrix mat) {
		return mat;
	}
	
	@Override
	public Matrix backProp(Matrix mat) {
		return mat;
	}
	
	
}
