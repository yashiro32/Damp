package en.menghui.android.damp.activations;

import Jama.Matrix;

public abstract class Activation {
	public abstract Matrix forwardProp(Matrix mat);
	public abstract Matrix backProp(Matrix mat);
}
