package en.menghui.android.damp.activations;

import en.menghui.android.damp.utils.NeuralNetUtils;
import Jama.Matrix;

public class TanhActivation extends Activation {
	@Override
	public Matrix forwardProp(Matrix mat) {
		return NeuralNetUtils.tanh(mat, false);
	}
	
	@Override
	public Matrix backProp(Matrix mat) {
		return NeuralNetUtils.tanh(mat, true);
	}
	
	
}
