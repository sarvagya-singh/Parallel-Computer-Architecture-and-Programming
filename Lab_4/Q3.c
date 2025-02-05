#include "mpi.h"
#include <stdio.h>

int main(int argc, char* argv[])
{
	int size,rank,key,i,j;
	int matrix[4][4];
	int res[4][4];
	int row[4],rowsum[4];

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	if(rank==0)
	{
		printf("Enter Matrix values: ");
		for(i=0;i<4;i++)
		{
			for(j=0;j<4;j++)
				scanf("%d",&matrix[i][j]);
		}
	}

	MPI_Scatter(matrix,4,MPI_INT,row,4,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Scan(&row,&rowsum,4,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
	MPI_Gather(&rowsum,4,MPI_INT,&res,4,MPI_INT,0,MPI_COMM_WORLD);

	if(rank==0)
	{
		printf("Resultant matrix is: \n");
		for(i=0;i<4;i++)
		{
			for(j=0;j<4;j++)
			{
				printf("%d  ",res[i][j]);
			}
			printf("\n");
		}
	}
	MPI_Finalize();
    return 0;
}