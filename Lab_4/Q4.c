#include <stdio.h>
#include <mpi.h>
#include<string.h>

int main(int argc, char *argv[]) {
    int rank, size;
    char str[40];
    char tmp[40];
    char res[100]; 
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;

    if (rank == 0) {
        printf("Enter the string to be modified: ");
        scanf("%s", str);
    }

    MPI_Bcast(str, 40, MPI_CHAR, 0, MPI_COMM_WORLD);

    int len = strlen(str);
    int count = len / size; 
    int remainder = len % size;

    int start = rank * count + (rank < remainder ? rank : remainder);
    int end = start + count + (rank < remainder ? 1 : 0);

    for (int i = start; i < end; i++) {
        tmp[i - start] = str[i];
        for (int j = 1; j < rank + 1; j++) {
            tmp[i - start + j] = str[i]; 
        }
    }

    if (rank != 0) {
        MPI_Send(tmp, end - start + rank, MPI_CHAR, 0, rank, MPI_COMM_WORLD);
    } else {
        strcpy(res, tmp); 

        for (int i = 1; i < size; i++) {
            int recv_count = count + (i < remainder ? 1 : 0) + i; 
            MPI_Recv(tmp, recv_count, MPI_CHAR, i, i, MPI_COMM_WORLD, &status);
            strncat(res, tmp, recv_count); 
        }

        printf("Modified string is: %s\n", res);
    }

    MPI_Finalize();
    return 0;
}
