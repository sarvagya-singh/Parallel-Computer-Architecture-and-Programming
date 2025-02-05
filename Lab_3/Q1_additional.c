#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    int rank, size, M, N;
    int *input_array = NULL, *local_array = NULL, *result_array = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the value of M (number of elements per process):\n");
        scanf("%d", &M);

        N = size;

        input_array = (int *)malloc(N * M * sizeof(int));
        result_array = (int *)malloc(N * M * sizeof(int));  
        printf("Enter the %d elements:\n", N * M);
        for (int i = 0; i < N * M; i++) {
            scanf("%d", &input_array[i]);
        }
    }
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    local_array = (int *)malloc(M * sizeof(int));

    MPI_Scatter(input_array, M, MPI_INT, local_array, M, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < M; i++) {
        local_array[i] = (int)pow(local_array[i], rank + 2);
    }
    MPI_Gather(local_array, M, MPI_INT, result_array, M, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Final Result Array: \n");
        for (int i = 0; i < N * M; i++) {
            printf("%d ", result_array[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
