#include <stdio.h>
#include <assert.h>
#include "spasm.h"

int main(int argc, char **argv) {
    assert(argc > 1);
    int field = atoi(argv[1]);
    char *gen = argv[2];
    char *par = argv[3];
    FILE *f = fopen(argv[4], "r");
    int get_par = atoi(argv[5]);
    spasm_triplet *T = spasm_load_sms(f, field);
    fclose(f);
    spasm *A = spasm_compress(T);
    spasm_triplet_free(T);
    spasm *A_t = spasm_transpose(A, SPASM_WITH_NUMERICAL_VALUES);
    printf("# computing kernel\n");
    spasm *K = spasm_kernel(A_t, SPASM_IDENTITY_PERMUTATION);
    FILE *fp = fopen(gen, "w+");
    spasm_save_csr(fp, K);
    fclose(fp);
    if (get_par == 0){
        K = spasm_kernel(K, SPASM_IDENTITY_PERMUTATION);
        fp = fopen(par, "w+");
        spasm_save_csr(fp, K);
        fclose(fp);
    }
    spasm_csr_free(A_t);
    spasm_csr_free(A);
    spasm_csr_free(K);
    printf("Done");
    return 0;
}