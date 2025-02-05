#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define TAG 0
#define WORD_SIZE 100
void toggle_case(char *word) {
    for (int i = 0; word[i] != '\0'; ++i) {
        if (islower(word[i]))
            word[i] = toupper(word[i]);
        else if (isupper(word[i]))
            word[i] = tolower(word[i]);
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    char word[WORD_SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            printf("This program requires exactly two processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) { 
        printf("Process 0: Enter a word to send: ");
        scanf("%s", word);

        MPI_Ssend(word, WORD_SIZE, MPI_CHAR, 1, TAG, MPI_COMM_WORLD);
        printf("Process 0: Sent word '%s' to Process 1.\n", word);

        MPI_Recv(word, WORD_SIZE, MPI_CHAR, 1, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 0: Received toggled word '%s' from Process 1.\n", word);
    } else if (rank == 1) { 
        MPI_Recv(word, WORD_SIZE, MPI_CHAR, 0, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1: Received word '%s' from Process 0.\n", word);

        toggle_case(word);

        MPI_Ssend(word, WORD_SIZE, MPI_CHAR, 0, TAG, MPI_COMM_WORLD);
        printf("Process 1: Sent toggled word '%s' back to Process 0.\n", word);
    }

    MPI_Finalize();
    return 0;
}
