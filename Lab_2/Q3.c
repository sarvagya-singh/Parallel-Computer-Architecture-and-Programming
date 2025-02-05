#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int square(int num) {
    return pow(num, 2);
}

int cube(int num) {
    return pow(num, 3);
}

int main(int argc, char* argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int a[size];
    int buffer_size = 100; 
    char *buffer = (char *)malloc(buffer_size);

    if (rank == 0) {
        MPI_Buffer_attach(buffer, buffer_size);
        for (int i = 1; i < size; i++) {
            printf("Enter value for position %d: ", i);
            scanf("%d", &a[i]);
            MPI_Bsend(&a[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        MPI_Buffer_detach(&buffer, &buffer_size);
    } else {
        int received_number;
        MPI_Recv(&received_number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank % 2 == 0) {
            printf("Square of the received element from Process %d is: %d\n", rank, square(received_number));
        } else {
            printf("Cube of the received element from Process %d is: %d\n", rank, cube(received_number));
        }
    }

    free(buffer); 
    MPI_Finalize();
    return 0;
}
