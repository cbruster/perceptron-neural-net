package linearAlgebra;

public class Vector extends Matrix {

	// A vector is truly a matrix with one column. This class is an explicit object
	// that allows for this sort of treatment. 
	
	public Vector(double[] arg) {
		super(arg.length, 1);
		for(int i = 0; i < arg.length; i++) {
			this.matrix[i][0] = arg[i];
		}
	}
	
	//initialize empty vector of given size
	public Vector(int size) {
		super(size, 1);
	}
	
	// the matrix field of a vector isn't meant to be used directly
	public double[] getComponents() {
		return this.getColumnAsArray(0);
	}
	
	//simple vector subtraction
	public static Vector Sub(Vector a, Vector b) 
	{
		double[] A = a.getComponents();
		double[] B = b.getComponents();
		if(A.length != B.length) 
			throw new RuntimeException("Cannot subtract vectors of unequal size.");
		for(int i = 0; i < A.length; i++) {
			A[i] -= B[i];
		}
		return new Vector(A);
	}
	
	public static Vector Add(Vector a, Vector b) {
		double[] A = a.getComponents();
		double[] B = b.getComponents();
		if(A.length != B.length) 
			throw new RuntimeException("Cannot add vectors of unequal size.");
		for(int i = 0; i < A.length; i++) {
			A[i] += B[i];
		}
		return new Vector(A);
		
	}
	
	public static Vector hadamard(Vector arg1, Vector arg2) 
	{
		double[] a = arg1.getComponents();
		double[] b = arg2.getComponents();
		if(a.length != b.length)
			throw new RuntimeException("Vectors must be of equal size.");
		double[] res = new double[a.length];
		for(int i = 0; i < res.length; i++) {
			res[i] = a[i] * b[i];
		}
		return new Vector(res);
	}
}