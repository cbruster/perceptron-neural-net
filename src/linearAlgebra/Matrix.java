package linearAlgebra;

import java.util.ArrayList;

public class Matrix {
	
	public final int rows;
	public final int cols;
	public final double[][] matrix;
	
	/* Constructors */
	
	public Matrix(double[][] data) {
		rows = data.length;
		cols = data[0].length;
		matrix = new double[rows][cols];
		
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				matrix[i][j] = data[i][j];
			}
		} 
	}
	public Matrix(int rows, int cols) { //build matrix object with an array, but no data in the array
		this.rows = rows;
		this.cols = cols;
		matrix = new double[rows][cols];
	}
	public Matrix(int size) { //constructor for specifically square matricies
		rows = size;
		cols = rows;
		matrix = new double[size][size];
	}
	public Matrix() { //null matrix object
		rows = 0;
		cols = 0;
		matrix = null;
	}
	
	/* Class Methods */
	
	//make an identity matrix
	public static Matrix eye(int size) {
		Matrix eye = new Matrix(size); //an identity matrix MUST be square
		for(int i = 0; i < size; i++) {
			eye.matrix[i][i] = 1;
		}
		return eye;
	}
	
	//fill a matrix with a given number
	public void fillMatrix(int fillnum) {
		for(int i = 0; i < this.rows; i++) {
			for(int j = 0; j < this.cols; j++) {
				this.matrix[i][j] = fillnum;
			}
		}
	}
	
	//fill a matrix with random values in domain [0, 1]. 
	public void fillMatrixRandom() {
		for(int i = 0; i < this.rows; i++) {
			for(int j = 0; j < this.cols; j++) {
				this.matrix[i][j] = Math.random();
			}
		}
	}
	//transpose a matrix
	public Matrix transpose() {
		Matrix res = new Matrix(cols, rows);
		for(int i = 0; i < cols; i++) {
			for (int j = 0; j < rows; j++) {
				res.matrix[i][j] = this.matrix[j][i];
			}
		}
		return res;
	}

	//simple matrix addition
	public Matrix mAdd(Matrix arg) {
		Matrix res = this;
		if ((res.rows != arg.rows) || (res.cols != arg.cols)) 
			throw new RuntimeException("Matrix dimensions are not equal.");
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				res.matrix[i][j] += arg.matrix[i][j];
			}
		}
		return res;
	}
	
	//simple matrix subtraction
	public Matrix mSub(Matrix arg) {
		Matrix res = this;
		if ((res.rows != arg.rows) || (res.cols != arg.cols)) 
			throw new RuntimeException("Matrix dimensions are not equal.");
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				res.matrix[i][j] -= arg.matrix[i][j];
			}
		}
		return res;
	}
	
	//print matrix to the console. WARNING: data printed this way is imprecise
	public void putMatrix() {
		for(int i = 0; i < this.rows; i++) {
			System.out.print("[");
			for(Double d : this.matrix[i]) System.out.printf("%5.2f", d);
			System.out.print(" ]\n");
		}
		System.out.println("");
	}
	
	//matrix scalar multiplication
	public void scale(double scalar) {
		for(int i = 0; i < this.rows; i++) {
			for(int j = 0; j < this.cols; j++) {
				this.matrix[i][j] *= scalar;
			}
		}
	}
	
	//matrix multiplication
	public Matrix multiply(Matrix arg) {
		Matrix res = new Matrix(this.rows, arg.cols);
		if (this.cols != arg.rows)
			throw new RuntimeException("Illegal matrix dimensions.");
		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < arg.cols; j++) { 
				for (int k = 0; k < arg.rows; k++) {
					res.matrix[i][j] += this.matrix[i][k] * arg.matrix[k][j];
				}
			}
		}
		return res;
	}
	
	//get a row in a matrix. Note: matrix rows are zero indexed
	public double[] getRow(int row) {
		double[] rowCopy = new double[this.matrix[row].length];
		rowCopy = this.matrix[row];
		
		return rowCopy;
	}
	
	//get a column in a matrix. Note matrix colums are zero indexed
	public double[] getColumnAsArray(int col) {
		double[] colArray = new double[this.rows];
		for(int i = 0; i < this.rows; i++) {
			colArray[i] = this.matrix[i][col];
		}
		return colArray;
	}
	
	//get a column in a matrix. Note this method returns a vector object
	public Vector getColumnAsVector(int col) {
		Vector colVector;
		double[] colArray = new double[this.rows];
		for(int i = 0; i < this.rows; i++) {
			colArray[i] = this.matrix[i][col];
		}
		colVector = new Vector(colArray);
		return colVector;
	}
	
	//get the determinant of a matrix by way of Row Echelon Form
	public static double det(Matrix arg) {
		
		Matrix workingObj = new Matrix(arg.matrix);
		
		if (arg.cols != arg.rows)
			throw new RuntimeException("Can't take determinant: Matrix must be square");
		
		double det = 1;
		double[][] data = workingObj.matrix;
		int size = arg.rows;

		boolean detIsNegative = false;
		
		for(int i = 0; i < size-1; i++) {
			
			workingObj = new Matrix(data);

			double pivot = data[i][i];
			
			if(pivot == 0) {
				double[] tmp = workingObj.getRow(i);
				workingObj.matrix[i] = workingObj.matrix[i+1];
				workingObj.matrix[i+1] = tmp;
				
				detIsNegative = !detIsNegative; //flip the sign of det
				data = workingObj.matrix;   
				pivot = data[i][i];
			}

			double[] row = workingObj.getRow(i);
			
			//elimination for current iteration is done here
			for(int j = (i+1); j < size; j++) {
				double mult = data[j][i] / pivot; //indexing eliminators
				for(int k = 0; k < row.length; k++) {
					data[j][k] -= (row[k] * mult);
				} 
			}
		}
		for(int i = 0; i < size; i++) {
			det *= data[i][i];
		}
		if(detIsNegative == true) {
			det = -det;
		}
		return det;
	}
	
	//return an inverted form of original object
	public static Matrix invertMatrix(Matrix arg) { 
		
		if (Matrix.det(arg) == 0) 
			throw new RuntimeException("Zero determinant: Cannot invert matrix.");
		
		int i, j, k;
		int N = arg.rows;
		int M = arg.cols;
		Matrix B = new Matrix(N, M);       //this will be the inverse matrix
		Matrix Z = new Matrix(N, M);       //Intermediary matrix
		Matrix C = Matrix.eye(N);          //Identity reference matrix
		Matrix L = Matrix.eye(N);          //Becomes L part of decomposition
		Matrix U;                          //Becomes U part of decomposition
		
		//upper triangular matrix is built first
		double[][] Udata = new Matrix(arg.matrix).matrix;
		ArrayList<Double> multList = new ArrayList<Double>(); //store multipliers for L
		
		for(i = 0; i < N - 1; i++) {

			double pivot = Udata[i][i];
			
			if(pivot == 0) {
				double[] tmp = Udata[i];
				Udata[i] = Udata[i+1];
				Udata[i+1] = tmp;
				pivot = Udata[i][i];
			}

			double[] row = Udata[i];
			
			//elimination for current iteration is done here
			for(j = (i+1); j < N; j++) {
				double mult = Udata[j][i] / pivot; //indexing eliminators
				multList.add(mult); //add multipliers to a list for building L.
				for(k = 0; k < row.length; k++) {
					Udata[j][k] -= (row[k] * mult);
				} 
			}
		}
		U = new Matrix(Udata); //U matrix is finished
		
		//Lower triangular matrix is built here
		Double[] multArray = multList.toArray(new Double[multList.size()]);
		k = 0; //index the multArray
		
		for(i = 0; i < N; i++) {
			for(j = (i+1); j < N; j++) {
				L.matrix[j][i] = multArray[k];
				k++;
			}
		} //L is finished.
		
		for(k = 0; k < B.cols; k++) { // LZ = C in first step, UX = Z in second. 
			
			//forward sub down the Z vector
			for(i = 0; i < Z.rows; i++) {
				Z.matrix[i][k] = C.matrix[i][k] - 
						dotProduct(
								L.getRow(i), 
								Z.getColumnAsArray(k));
			}
			//Back-sub is done up B vector, subtracting U dot product with B from Z, then dividing out B coeff.
			for(i = (B.cols-1); i >= 0; i--) {
				B.matrix[i][k] = (Z.matrix[i][k] - 
						dotProduct(
								U.getRow(i), 
								B.getColumnAsArray(k))) / U.matrix[i][i];
			}
		}
		return B;
	}
	
	//take two subarrays of a matrix and dot them. Typically dotting a row with a column, but not necessarily.
	public static double dotProduct(double[] row, double[] col) {
		if(row.length != col.length)
			throw new RuntimeException("Cannot take dotProduct: Input arrays are inequal length.");
		double dotProd = 0;
		for(int i = 0; i < row.length; i++) {
			dotProd += row[i] * col[i];
		}
		return dotProd;
	}
	
	public static double dotProduct(Matrix a, Matrix b) {
		if(a.rows != b.cols)
			throw new RuntimeException("Cannot take dotProduct: Illegal matrix dimensions.");
		double dotProd = 0.0d;
		Matrix res = a.multiply(b);
		for(int i = 0; i < res.rows; i++) {
			for(int j = 0; j < res.cols; j++) {
				dotProd += res.matrix[i][j];
			}
		}
		return dotProd;
	}
	
	public static Matrix vectorsToMatrix(Vector[] vectorArray) {
		Matrix mat = new Matrix(vectorArray[0].rows, vectorArray.length); //matrix is determined based on element 0 dimensions
		
		for(int i = 0; i < mat.cols; i++) {
			
			double[] vect = vectorArray[i].getComponents(); //get the vector going in column i
			
			for(int j = 0; j < mat.rows; j++) {
				mat.matrix[j][i] = vect[j];
			}
		}
		
		return mat;
	}
}
