#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_LOW(rank, np, n) ((rank) * (n) / (np))
#define BLOCK_HIGH(rank, np, n) (BLOCK_LOW((rank) + 1, np, n) - 1)
#define BLOCK_SIZE(rank, np, n) \
    (BLOCK_HIGH(rank, np, n) - BLOCK_LOW(rank, np, n) + 1)
#define BLOCK_OWNER(index, np, n) ((np) * (((index) + 1) / (n)))

// use 4bit msb to store the new state
const unsigned MASK = 0xF;
const unsigned SHIFT = 4;
#define SET_LIVE(x)          \
    do {                     \
        (x) |= 1 << (SHIFT); \
    } while (0)
#define SET_DEAD(x)             \
    do {                        \
        (x) &= ~(1 << (SHIFT)); \
    } while (0)
#define UPDATE(x, liveCount)                            \
    do {                                                \
        if ((x)&1) {                                    \
            if ((liveCount) == 2 || (liveCount) == 3) { \
                SET_LIVE((x));                          \
            } else {                                    \
                SET_DEAD((x));                          \
            }                                           \
        } else {                                        \
            if ((liveCount) == 3) {                     \
                SET_LIVE((x));                          \
            } else {                                    \
                SET_DEAD((x));                          \
            }                                           \
        }                                               \
    } while (0)

void readRowMajorMatrix(char* fileName, char** array, int* M, int* N) {
    FILE* inFile = NULL;

    inFile = fopen(fileName, "r");
    if (inFile == NULL) {
        exit(1);
        return;
    }
    fread(M, sizeof(int), 1, inFile);
    fread(N, sizeof(int), 1, inFile);

    int chunkSize = (*M) * (*N);
    // +2N to store a row before and a row after the block
    *array = malloc((chunkSize + 2 * (*N)) * sizeof(char));

    // +*N to reserve an empty row
    memset(*array, 0, *N * sizeof(char));
    memset(*array + (*M + 1) * (*N), 0, *N * sizeof(char));
    fread(*array + *N, sizeof(char), chunkSize, inFile);
    fclose(inFile);
}

char** reshapeTo2D(char* array, int M, int N) {
    char** reshaped = malloc(M * sizeof(char*));
    for (int i = 0; i < M; ++i) {
        reshaped[i] = array;
        array += N;
    }
    return reshaped;
}

int main(int argc, char* argv[]) {
    assert(argc == 3);
    const int ITERATIONS = atoi(argv[1]);
    const int PRINT_CYCLE = atoi(argv[2]);

    int M = 0, N = 0;
    char* flattenedArray = NULL;
    readRowMajorMatrix("state", &flattenedArray, &M, &N);

    // +2 for row before and after block
    char** array = reshapeTo2D(flattenedArray, M + 2, N);

    for (int i = 0; i < ITERATIONS; ++i) {
        // calculate new state of each cell
        for (int r = 1; r <= M; ++r) {
            int liveCount = (array[r - 1][0] + array[r - 1][1] + array[r][0] +
                             array[r][1] + array[r + 1][0] + array[r + 1][1]) &
                            MASK;
            UPDATE(array[r][0], liveCount - (array[r][0] & MASK));
            for (int c = 1; c < N - 1; ++c) {
                // add col
                liveCount += (array[r - 1][c + 1] + array[r][c + 1] +
                              array[r + 1][c + 1]) &
                             MASK;
                UPDATE(array[r][c], liveCount - (array[r][c] & MASK));
                // remove col
                liveCount -= (array[r - 1][c - 1] + array[r][c - 1] +
                              array[r + 1][c - 1]) &
                             MASK;
            }
            UPDATE(array[r][N - 1], liveCount - (array[r][N - 1] & MASK));
        }
        for (int r = 1; r <= M; ++r) {
            for (int c = 0; c < N; ++c) {
                // set to new state
                array[r][c] = ((array[r][c] & (1 << SHIFT)) >> SHIFT) | '0';
            }
        }
        if (i % PRINT_CYCLE == 0) {
            for (int r = 1; r <= M; ++r) {
                fwrite(array[r], 1, N, stdout);
                putchar('\n');
            }
            putchar('\n');
            fflush(stdout);
        }
    }

    free(array);
    free(flattenedArray);
    return 0;
}