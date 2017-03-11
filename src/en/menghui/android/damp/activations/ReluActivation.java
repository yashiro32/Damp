package en.menghui.android.damp.activations;

import en.menghui.android.damp.utils.NeuralNetUtils;
import Jama.Matrix;

public class ReluActivation extends Activation {
	@Override
	public Matrix forwardProp(Matrix mat) {
		return NeuralNetUtils.relu(mat, false);
	}
	
	@Override
	public Matrix backProp(Matrix mat) {
		return NeuralNetUtils.relu(mat, true);
	}
	
	
}
