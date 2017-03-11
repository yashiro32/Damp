package en.menghui.android.damp.utils;

public class MathUtils {
	public static int intMax(int a, int b) {
		if (a >= b) {
			return a;
		} else {
			return b;
		}
	}
	
	public static int intMin(int a, int b) {
		if (a <= b) {
			return a;
		} else {
			return b;
		}
	}
	
	public static int getBinomial(int n, double p) {
		int x = 0;
		for(int i = 0; i < n; i++) {
			if(Math.random() < p)
				x++;
		}
		
		return x;
	}
}
