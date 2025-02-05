#include "mpi.h"
#include <stdio.h>
 
void ErrorHandler(int ecode, const char *error_desc)
{
    if (ecode != MPI_SUCCESS)
    {
        char err_str[BUFSIZ];
        int strlen, err_class;
 
        MPI_Error_class(ecode, &err_class);          
        MPI_Error_string(err_class, err_str, &strlen);   
 
        printf("Error Description: %s\n", error_desc);
        printf("MPI Error Class: %d\n", err_class); 
        printf("MPI Error Message: %s\n", err_str);  
    }
}
 
int factorial(int n) {
    int fact = 1;
    for (int i = 1; i <= n; i++) {
        fact *= i;
    }
    return fact;
}
 
int main(int argc, char *argv[])
{
    int rank, size, fact, factsum, total_factsum;
    int ecode;
 
    ecode = MPI_Init(&argc, &argv); 
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Init failed");
        MPI_Finalize();
        return -1;
    }
 
    ecode = MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Errhandler_set failed");
        MPI_Finalize();
        return -1;
    }
 
    ecode = MPI_Comm_rank(MPI_COMM_NULL, &rank);
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Comm_rank failed with MPI_COMM_NULL");
    }
 
    ecode = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Comm_rank failed");
        MPI_Finalize();
        return -1;
    }
 
    ecode = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Comm_size failed");
        MPI_Finalize();
        return -1;
    }
 
    int send_data = 42;
    float recv_data; 
    ecode = MPI_Send(&send_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Send failed");
    }
 
    ecode = MPI_Recv(&recv_data, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Recv failed with mismatched type");
    }
 
    ecode = MPI_Send(&send_data, 1, MPI_INT, size, 0, MPI_COMM_WORLD); 
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Send failed with invalid rank");
    }
 
    fact = factorial(rank + 1);
    printf("Rank %d: Factorial of %d = %d\n", rank, rank + 1, fact);
 
    ecode = MPI_Scan(&fact, &factsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Scan failed");
        MPI_Finalize();
        return -1;
    }
 
    printf("Rank %d: Factorial = %d, Cumulative Sum of Factorials = %d\n", rank, fact, factsum);
 
    ecode = MPI_Finalize();
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Finalize failed");
        return -1;
    }
 
    return 0;
}