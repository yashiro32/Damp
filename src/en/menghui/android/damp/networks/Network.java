package en.menghui.android.damp.networks;

import java.util.ArrayList;
import java.util.List;

import en.menghui.android.damp.layers.FullyConnectedLayer;
import en.menghui.android.damp.layers.Layer;
import en.menghui.android.damp.layers.SoftmaxLayer;
import Jama.Matrix;

public class Network {
	public String name = "";
	
	public List<Layer> layers = new ArrayList<Layer>();
	
	public int epochs = 0; 
}
