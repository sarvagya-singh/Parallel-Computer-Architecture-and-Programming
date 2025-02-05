#include <stdio.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    char str[] = "Hello";
    int str_len = strlen(str);

    
    if (rank < str_len) {
        if (str[rank] >= 'A' && str[rank] <= 'Z') {
        str[rank] = str[rank] + 32;  
    } else if (str[rank] >= 'a' && str[rank] <= 'z') {
        str[rank] = str[rank] - 32;  
    }
    }

    printf("String has been edited to %s by Process %d\n", str, rank);

    MPI_Finalize();
    return 0;
}
