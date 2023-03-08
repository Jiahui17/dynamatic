#include <stdlib.h>
#include "test_memory_11.h"

#define AMOUNT_OF_TEST 1

void test_memory_11(int a[4]) {
	int x;
	for (int i = 1; i < 4; i++) {
		a[i] = a[i-1] + a[i];

	}
	for (int i = 1; i < 4; i++) {
		a[i] = a[i-1];

	}
}

int main(void){
	inout_int_t a[AMOUNT_OF_TEST][4];
	in_int_t n[AMOUNT_OF_TEST];

	srand(13);
	for(int i = 0; i < AMOUNT_OF_TEST; ++i){
		n[i] = 3;
		for(int j = 0; j < 4; ++j){
			a[i][j] = (rand() % 100) - 50;
		}
	}
    
	//for(int i = 0; i < AMOUNT_OF_TEST; ++i){
	int i = 0;
	test_memory_11(a[i]);
	//}
}

