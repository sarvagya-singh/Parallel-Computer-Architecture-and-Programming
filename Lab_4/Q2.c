#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
 
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, nop, search;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nop);
    MPI_Status stat;
    int mat[3][3];
    int temp[3];
    if(rank == 0) {
        printf("Enter the 9 elements:\n");
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                scanf("%d",&mat[i][j]);
            }
        }
        printf("Enter the number to be searched:\n");
        scanf("%d",&search);
    }
    MPI_Bcast(&search, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(mat, 3, MPI_INT, temp, 3, MPI_INT, 0, MPI_COMM_WORLD);
    int occ = 0, sum_occ = 0;
    for(int i = 0; i < 3; i++) {
        if(search == temp[i]) {
            occ++;
        }
    }
    MPI_Reduce(&occ, &sum_occ, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0) {
        printf("Total Number of Occurences: %d\n", sum_occ);
    }
    MPI_Finalize();
    exit(0);
}