#include "mpi.h"
#include <stdio.h>
#include <math.h>

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int value;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) {
        printf("Enter an integer value: ");
        scanf("%d", &value);
    }
    int received_value;

    if (rank == 0) {
        MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&received_value, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Final value received by root: %d\n", received_value);
    } else if (rank == size - 1) {
        MPI_Recv(&received_value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        received_value++;
        MPI_Send(&received_value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(&received_value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        received_value++;
        MPI_Send(&received_value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}