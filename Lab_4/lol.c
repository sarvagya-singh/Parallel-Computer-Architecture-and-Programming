#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    int errcode,rank;
    MPI_Comm comm_null = MPI_COMM_NULL;
    errcode = MPI_Comm_rank(comm_null, &rank);

    if (errcode != MPI_SUCCESS) {
        char error_string[128];
        int length_of_error_string;
        int errorclass;
        MPI_Error_class(errcode, &errorclass);
        fprintf(stderr, "Error class code: %d\n", errorclass);

        // Get the error string
        MPI_Error_string(errorclass, error_string, &length_of_error_string);
        fprintf(stderr, "Error caught: %s\n", error_string);
        
    }
    MPI_Finalize();
    return 0;
}

