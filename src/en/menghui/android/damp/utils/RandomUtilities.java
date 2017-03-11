package en.menghui.android.damp.utils;

import java.util.Calendar;
import java.util.Date;
import java.util.Random;

public class RandomUtilities {
	
	private static Random random = new Random(seed());
	private static double val;
	private static boolean returnVal;
	
	public static long seed() {
		Calendar calendar = Calendar.getInstance();    
		calendar.set(Calendar.MILLISECOND, 0); // Clear the millis part. Silly API.
		calendar.set(2010, 8, 14, 0, 0, 0); // Note that months are 0-based
		Date date = calendar.getTime();
		long millis = date.getTime(); // Millis since Unix epoch
		
		return millis;
	}
	
	public static double gaussianRandom() {
		if (returnVal) {
			returnVal = false;
			return val;
		}
		
		double u = 2 * random.nextDouble() - 1;
		double v = 2 * random.nextDouble() - 1;
		double r = u*u + v*v;
		
		if (r == 0 || r > 1) {
			return gaussianRandom();
		}
		
		double c = Math.sqrt(-2*Math.log(r)/r);
		val = v*c;
		returnVal = true;
		
		return u*c;
	}
	
	public static double randn(double mu, double std) {
		return mu + gaussianRandom()*std;
	}
}
