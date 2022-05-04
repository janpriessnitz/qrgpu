#include <cstdio>
#include <ctime>
#include <cstdlib>

#include "Matrix.h"
#include "QRTuning.h"

int main(int argc, char** argv) {

    if (argc < 4) {
        fprintf(stderr, "Need 3 arguments - matrix rows, cols, output tuning params filename\n");
        return 1;
    }
    int rows = strtol(argv[1], nullptr, 10);
    int cols = strtol(argv[2], nullptr, 10);
    std::string paramsFile = argv[3];
    if (rows <= 0) {
        fprintf(stderr, "Invalid first argument\n");
        return 1;
    }
    if (cols <= 0) {
        fprintf(stderr, "Invalid second argument\n");
        return 1;
    }

    if (rows < cols) {
        fprintf(stderr, "Invalid matrix size - number of rows must be greater or equal than number of cols\n");
        return 1;
    }

    // srand(time(nullptr));
    srand(2);

    QRTuning(rows, cols, paramsFile);
    return 0;
}
