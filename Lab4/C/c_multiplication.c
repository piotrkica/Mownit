#include <stdio.h>
#include <time.h>
#include <sys/times.h>
#include <unistd.h>
#include <gsl/gsl_blas.h>
#include <stdlib.h>

long double gettime(clock_t t1, clock_t t2){
    return ((long double)(t2 - t1) / sysconf(_SC_CLK_TCK));
}


void naive_multiplication(int **matrixA, int **matrixB, int n){
    int* matrix[n];
    for (int i = 0; i < n; i++){
        matrix[i] = calloc(n, sizeof(int));
        for (int j = 0; j < n; j++){
            matrix[i][j] = 0;
        }
    }
    for (int k = 0; k < n; k++){
        for (int j = 0; j < n; j++){  
            for (int i = 0; i < n; i++){  
                matrix[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    } 
}

void better_multiplication(int** matrixA, int** matrixB, int n){
    int* matrix[n];
    for (int i = 0; i < n; i++){
        matrix[i] = calloc(n, sizeof(int));
        for (int j = 0; j < n; j++){
            matrix[i][j] = 0;
        }
    }
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){  
            for (int k = 0; k < n; k++){
                matrix[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
}

void blas_multiplication(double* matrixA, double* matrixB, int n){
    double* matrix = malloc(sizeof(double) * n * n);
    gsl_matrix_view A = gsl_matrix_view_array(matrixA, n, n);
    gsl_matrix_view B = gsl_matrix_view_array(matrixB, n, n);
    gsl_matrix_view C = gsl_matrix_view_array(matrix, n, n);
    gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,
                    1.0, &A.matrix, &B.matrix,
                    0.0, &C.matrix);
}

int** create_random_matrix(int x, int y){
    int ** matrix = calloc(x, sizeof(int*));

    for (int i = 0; i < x; i ++){
        int * values = calloc(y, sizeof(int));
        matrix[i] = values;
        for (int j = 0; j < y; j ++){
            values[j] = rand() % 100;
        }
    }
    return matrix;
}

double* create_random_blas_matrix(int x, int y){
    double* matrix = (double *)malloc(x*y*sizeof(double));
    for (int i = 0; i < (x*y); i++) {
        matrix[i] = (double) (rand() % 100);
    }
    return matrix;
}

int main(){
    srand(time(NULL));
    struct tms start_tms;
    struct tms end_tms;
    clock_t clock_start;
    clock_t clock_end;

    FILE *f = fopen("results.csv", "w+");

    for (int n = 10; n <= 400; n += 10){
        for (int i = 0; i < 10; i++){
            int ** matrixA = create_random_matrix(n, n);
            int ** matrixB = create_random_matrix(n, n);
            double *blasMatrixA = create_random_blas_matrix(n,n);
            double *blasMatrixB = create_random_blas_matrix(n,n);

            fprintf(f, "%i,%i,", n, i);

            clock_start = times(&start_tms);
            naive_multiplication(matrixA, matrixB, n);
            clock_end = times(&end_tms);
            fprintf(f, "%Lf,", gettime(clock_start, clock_end));

            clock_start = times(&start_tms);
            better_multiplication(matrixA, matrixB, n);
            clock_end = times(&end_tms);
            fprintf(f, "%Lf,", gettime(clock_start, clock_end));

            clock_start = times(&start_tms);
            blas_multiplication(blasMatrixA, blasMatrixB, n);
            clock_end = times(&end_tms);
            fprintf(f, "%Lf\n", gettime(clock_start, clock_end));
        }
    }
    
    fclose(f);
    return 0;
}