#include <stdio.h>
#include "mpi.h"

int Factorial(int input){
	if (input == 0){
		return 1;
	}
	else{
		return input * Factorial(input-1);
	}
}

int Fibonacci(int input)
{
	if (input==1 || input == 2){
		return 1;
	}
	else if (input == 0)
	{
		return 0;
	}
	else{
		return Fibonacci(input-1) + Fibonacci(input-2);
	}
}

int main(int argc, char** argv){
	int rank, size;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);

	if (rank%2 == 0) {
		int factorial = Factorial(rank);
		printf("Process %d has Factorial : %d\n",rank, factorial);
	}
	else{
		int fib_num = Fibonacci(rank);
		printf("Process %d has Fibonacci Number : %d\n",rank, fib_num);
	}
	MPI_Finalize();
	return 0;
}