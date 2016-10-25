package en.menghui.android.damp.utils;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import com.opencsv.CSVReader;

import android.content.Context;

public class FileUtils {
	public static List<String[]> readCSV(String filename, Context context, String delimiter) {
		BufferedReader br = null;
        String line = "";
        
        List<String[]> list = new ArrayList<String[]>();
        
		try {
			/* InputStreamReader is = new InputStreamReader(context.getAssets().open(filename));
			
			br = new BufferedReader(is);
			
			while((line = br.readLine()) != null) {
				// Use comma as separator.
				String[] values = line.split(delimiter);
				
				list.add(values);
			}
			
			br.close();
			is.close(); */
			
			CSVReader reader = new CSVReader(new InputStreamReader(context.getAssets().open(filename)));
            while(true) {
                String[] next = reader.readNext();
                if(next != null) {
                    list.add(next);
                } else {
                    break;
                }
            }
            
            reader.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return list;
	}
	
	
}
