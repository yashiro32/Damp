package en.menghui.android.damp.activations;

import en.menghui.android.damp.utils.NeuralNetUtils;
import Jama.Matrix;

public class SigmoidActivation extends Activation {
	@Override
	public Matrix forwardProp(Matrix mat) {
		return NeuralNetUtils.sigmoid(mat, false);
	}
	
	@Override
	public Matrix backProp(Matrix mat) {
		return NeuralNetUtils.sigmoid(mat, true);
	}
	
	
}
