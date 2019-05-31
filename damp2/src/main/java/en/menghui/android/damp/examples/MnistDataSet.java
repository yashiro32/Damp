package en.menghui.android.damp.examples;

import java.util.Arrays;
import java.util.List;

import Jama.Matrix;
import android.content.Context;
import en.menghui.android.damp.utils.FileUtils;

public class MnistDataSet {
	private static final String TAG = "MNIST Data Set";
	public Matrix featuresMatrix;
	public Matrix labelsMatrix;
	
	public Context context;
	
	public MnistDataSet(Context context) {
		this.context = context;
	}
	
	public void loadDataSet(int labelColumn, int[] columnsToIgnore, boolean removeHeader) {
		List<String[]> list = FileUtils.readCSV("train_mnist_100rows.csv", context, ",");
		
		if (removeHeader) {
			list.remove(0); // Remove header at index 0.
		}
		
		listToMatrix(list, labelColumn, columnsToIgnore);
	}
	
	public void listToMatrix(List<String[]> list, int labelColumn, int[] columnsToIgnore) {
		for (int x = 0; x < list.size(); x++) {
			if (x == 0) {
				featuresMatrix = new Matrix(list.size(), list.get(x).length-columnsToIgnore.length-1);
				labelsMatrix = new Matrix(list.size(), 1);
				// continue;
			}
			
			// Use comma as separator.
			String[] values = list.get(x);
			
			int c = 0;
			for (int i = 0; i < values.length; i++) {
				// Log.i(TAG, "i: " + i);
				/* Arrays.sort(columnsToIgnore);
				if (containsValue(columnsToIgnore, i)) {
				    continue;
				} */
				
				if (i == labelColumn) {
					labelsMatrix.set(x, 0, Double.parseDouble(values[i]));
					continue;
				}
				
				featuresMatrix.set(x, c, Double.parseDouble(values[i])/255.0);
				
				c++;
			}
		}
	}
	
	private boolean containsValue(int[] arr, int value) {
		boolean contains = false;
		
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] == value) {
				contains = true;
				break;
			}
		}
		
		return contains;
	}
	
	
}
