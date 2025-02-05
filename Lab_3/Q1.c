#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char **argv){
    int rank, size, *arr, x, ans, fact = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0){
        arr = (int *)malloc(size * sizeof(int));
        printf("enter the numbers\n");
        for (int i = 0; i < size; ++i){
            scanf("%d", &arr[i]);
        }
        for(int i = 2; i < arr[0] + 1; ++i){
            fact *= i;
        }
    }
    MPI_Scatter(arr, 1, MPI_INT, &x, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank > 0){
        for(int i = 2; i < x + 1; ++i){
            fact *= i;
        }
        printf("%d from rank %d\n", fact, rank);
    }
    MPI_Reduce(&fact, &ans, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0){
        printf("%d is the answer\n", ans);
    }
    MPI_Finalize();
}