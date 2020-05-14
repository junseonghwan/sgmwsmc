package common.util;

import java.io.File;
import java.io.PrintWriter;
import java.util.List;

import briefj.BriefIO;

public class OutputHelper 
{
	public static void writeLines(File file, List<String> contents)
	{
		PrintWriter writer = BriefIO.output(file);
		for (String line : contents)
			writer.println(line);
		writer.close();
	}

	public static void writeLinesOfDoubleArr(File file, String [] header, List<double []> contents)
	{
		PrintWriter writer = BriefIO.output(file);
		for (int i  = 0; i < header.length; i++)
		{
			writer.print(header[i]);
			if (i < header.length - 1) writer.print(",");
			else writer.println();
		}
		for (double [] row : contents)
		{
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < row.length; i++)
			{
				sb.append(row[i] + "");
				if (i < row.length - 1)
					sb.append(", ");
			}
			writer.println(sb.toString());
		}
		writer.close();
	}

	/**
	 * Helper function to write a vector of data to file as a column vector
	 * 
	 * @param filepath
	 * @param data
	 * @return
	 */
	public static <T> void writeVector(File file, List<T> data)
	{
		PrintWriter writer = BriefIO.output(file);
		for (T val : data)
		{
			writer.println(val.toString());
		}
		writer.close();
	}

	public static void writeVector(String filepath, double [] data)
	{
		PrintWriter writer = BriefIO.output(new File(filepath));
		for (double val : data)
		{
			writer.println(val);
		}
		writer.close();
	}

	public static void writeVector(File file, double [] data)
	{
		PrintWriter writer = BriefIO.output(file);
		for (double val : data)
		{
			writer.println(val);
		}
		writer.close();
	}

	public static <T> void writeTableAsCSV(File file, int numColumns, List<List<T>> data)
	{
		PrintWriter writer = BriefIO.output(file);
		for (List<T> d : data)
		{
			if (d.size() != numColumns)
				throw new RuntimeException("number of columns != actual data size");
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < numColumns; i++)
			{
				if (i == numColumns - 1)
					sb.append(d.get(i).toString());
				else
					sb.append(d.get(i).toString() + ", ");
			}
			writer.println(sb.toString());
		}
		writer.close();
	}
	
	public static void write2DArrayAsCSV(File file, String [] header, double [][] data)
	{
		PrintWriter writer = BriefIO.output(file);
		if (header != null) {
			if (header.length != data[0].length)
				throw new RuntimeException("Number of headers != number of columns in the data!");
			for (int i = 0; i < header.length; i++) {
				writer.print(header[i]);
				if (i < header.length - 1) writer.print(",");
				else writer.println();
			}
		}

		for (double [] row : data)
		{
			for (int j = 0; j < row.length; j++) 
			{
				writer.print(row[j]);
				if (j < row.length - 1)
					writer.print(",");
				else
					writer.println();
			}
		}
		writer.close();
	}

	public static void writeTableAsCSV(File file, String [] header, List<double []> data)
	{
		PrintWriter writer = BriefIO.output(file);
		if (header != null) {
			for (int i = 0; i < header.length; i++) {
				writer.print(header[i]);
				if (i < header.length - 1) writer.print(",");
				else writer.println();
			}
		}

		for (double [] row : data)
		{
			if (header != null && row.length != header.length)
				throw new RuntimeException("Not all rows have the same numbre of columns!");
			
			for (int j = 0; j < row.length; j++) 
			{
				writer.print(row[j]);
				if (j < row.length - 1)
					writer.print(",");
				else
					writer.println();
			}
		}
		writer.close();
	}

	public static void writeTableAsCSV(File file, String [] header, double []... data)
	{
		if (header.length != data.length)
			throw new RuntimeException("The number of headers != number of columns in the data");
		
		PrintWriter writer = BriefIO.output(file);
		for (int i = 0; i < header.length; i++) {
			writer.print(header[i]);
			if (i < header.length - 1) writer.print(",");
			else writer.println();
		}
		for (int i = 0; i < data[0].length; i++)
		{
			for (int j = 0; j < data.length; j++) 
			{
				writer.print(data[j][i]);
				if (j < data.length - 1)
					writer.print(",");
				else
					writer.println();
			}
		}
		writer.close();
	}

}
