#include <stdlib.h>
#include "loop_array.h"

#define AMOUNT_OF_TEST 1


void loop_array(in_int_t n, in_int_t k, inout_int_t c[LOOP_BOUND]) {

	for (int i = 1; i < n; i++) {
		c[i] = k + c[i-1];
	}
}

int main(void){
	in_int_t k[AMOUNT_OF_TEST];
	in_int_t n[AMOUNT_OF_TEST];
	inout_int_t c[AMOUNT_OF_TEST][LOOP_BOUND];

	srand(13);
	for(int i = 0; i < AMOUNT_OF_TEST; ++i){
		k[i] = rand() % LOOP_BOUND;
		n[i] = rand() % LOOP_BOUND;
		for(int j = 0; j < LOOP_BOUND; ++j){
			c[i][j] = 0;
		}
	}


	//for(int i = 0; i < AMOUNT_OF_TEST; ++i){
	int i = 0;
	loop_array(n[i], k[i], c[i]);
	//}
}

