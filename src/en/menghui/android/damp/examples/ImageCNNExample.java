package en.menghui.android.damp.examples;

import java.util.ArrayList;
import java.util.List;

import en.menghui.android.damp.R;
import en.menghui.android.damp.layers.ConvLayer;
import en.menghui.android.damp.optimizations.AdamOptimizer;
import en.menghui.android.damp.utils.Volume;
import android.app.Activity;
import android.os.Bundle;
import android.util.Log;

public class ImageCNNExample extends Activity {
	private static final String TAG = "Image CNN Example";
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		
		ConvLayer layer = new ConvLayer(5, 5, 8);
		layer.optimizer = new AdamOptimizer(0.9, 0.999, 0.01);
		layer.init(28, 28, 3);
		List<Volume> list = new ArrayList<Volume>();
		Volume volume = new Volume(28, 28, 3);
		list.add(volume);
		Volume volume2 = new Volume(28, 28, 3);
		list.add(volume2);
		
		Log.d(TAG, "Training starts");
		Log.d(TAG, "Input length: " + list.get(0).weights.length);
		long timeStart = System.currentTimeMillis();
		List<Volume> out = layer.forwardProp(list);
		Log.d(TAG, "Output length: " + out.get(0).weights.length);
		layer.backProp();
		layer.optimize();
		Log.d(TAG, "Training ends at " + (System.currentTimeMillis() - timeStart));
	}
}
