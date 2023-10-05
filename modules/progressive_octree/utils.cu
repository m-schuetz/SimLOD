
#include "utils.h.cu"

void printNumber(int64_t number, int leftPad){
	int digits = ceil(log10(double(number)));
	digits = (digits > 0) ? digits : 1;

	{
		int64_t digitCheck = 1;
		for(int i = 0; i < digits; i++){
			digitCheck = digitCheck * 10;
		}

		if(digitCheck == number){
			digits++;
		}
	}

	int numSeparators = (digits - 1) / 3;
	int stringSize = digits + numSeparators;

	for(int i = leftPad; i > stringSize; i--){
		printf(" ");
	}

	for(int digit = digits - 1; digit >= 0; digit--){

		int64_t a = pow(10.0, double(digit) + 1);
		int64_t b = pow(10.0, double(digit));

		int64_t tmp = number / a;
		int64_t tmp2 = number - (a * tmp);
		int64_t current = tmp2 / b;

		printf("%i", current);

		if((digit % 3) == 0 && digit > 0){
			printf("'");
		}

	}
}