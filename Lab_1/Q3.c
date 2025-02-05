#include "mpi.h"
#include<stdio.h>
int main(int argc, char *argv[]){
	int rank;
	int size;
	double num1=5.0;
	double num2 =3.0;
	double result;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	switch(rank){
		case 0:
			result = num1 + num2;
            printf("Process %d: Addition: %.2f + %.2f = %.2f\n", rank, num1, num2, result);
            break;
        case 1:
            result = num1 - num2;
            printf("Process %d: Subtraction: %.2f - %.2f = %.2f\n", rank, num1, num2, result);
            break;
        case 2:
            result = num1 * num2;
            printf("Process %d: Multiplication: %.2f * %.2f = %.2f\n", rank, num1, num2, result);
            break;
        case 3:
            
            if (num2 != 0.0) {
                result = num1 / num2;
                printf("Process %d: Division: %.2f / %.2f = %.2f\n", rank, num1, num2, result);
            } else {
                printf("Process %d: Division: Error! Division by zero.\n", rank);
            }
            break;
        default:
            printf("Process %d: No operation assigned.\n", rank);
            break;
    }

    MPI_Finalize();

    return 0;

	} 
