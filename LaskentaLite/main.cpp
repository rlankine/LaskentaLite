
#include "LaskentaLite.h"

#include <iomanip>

void demo();
bool test();

int main() try
{
	demo(); return 0;

	if (test())
	{
		cout << endl << "OK." << endl;
		return EXIT_SUCCESS;
	}
	else
	{
		cout << endl << "Fail!!" << endl;
		return EXIT_FAILURE;
	}
}
catch (char const *p)
{
	cout << endl << p << endl;
	return EXIT_FAILURE;
}
catch (...)
{
	cout << endl << "Diva tantrum!!!!" << endl;
}

// Demo code follows:

Expression Distance(Expression x0, Expression y0, Expression x1, Expression y1)
{
	Expression dx = x1 - x0;
	Expression dy = y1 - y0;
	return sqrt(dx * dx + dy * dy);
}

void demo()
{
	Variable t; t.Name("t");

	// Emulate two objects moving along parametric tracks '(t,t)' and '(sin(t),cos(t))'
	// I.e. one object moves along the diagonal line and the other along the unit circle
	// Compute minimal distance and the instant 't'

	Expression x0 = t;
	Expression y0 = t;
	Expression x1 = sin(t);
	Expression y1 = cos(t);

	// Note: To evaluate an Expression or a Variable, use explicit cast to double().

	Expression model = Distance(x0, y0, x1, y1);

	// When some value is assigned to 't', the value of 'model' computes the corresponding distance:
	cout << endl << "Model: " << model << endl << endl;

	for (int i = 0; i <= 10; ++i)
	{
		t = i / 10.0;
		cout << "t = " << double(t) << ", model = " << double(model) << endl;
	}

	Expression slope = model.Derive(t);

	// Each root of 'slope' indicates some local extremum of 'model' (later on we'll assume that
	// the first extremum in the positive direction starting from 't=0' is the global minimum)
	cout << endl << "Slope: " << slope << endl << endl;

	for (int i = 0; i <= 10; ++i)
	{
		t = i / 10.0;
		cout << "t = " << double(t) << ", slope = " << double(slope) << endl;
	}

	// Find first root of the 'slope' where 't>0', using 3rd order Householder's method
	int const N = 3;  // (N = 1 for Newton's method and N = 2 for Halley's method, but both diverge here)
	auto diffs = (1 / slope).Derive(t, N);
	Expression delta = N * diffs[N-1] / diffs[N];  // See https://en.wikipedia.org/wiki/Householder%27s_method

	cout << endl;  // long print: cout << endl << "Delta: " << delta << endl << endl;

	// Binds expression 't + delta' to variable 't'
	// Every invocation of Commit() evaluates the expression and assigns the result to the variable
	Bindings iterator;
	iterator.Insert(t, t + delta);

	// Iterate fixed number of times (in real application some end criterion is used instead)
	for (int i = 0; i < 10; ++i)
	{
		iterator.Commit();
		cout << "iteration " << i << ": t = " << double(t) << ", model = " << double(model) << endl;
	}

	cout << endl << "Result: Minimal distance is " << double(model) << " at time t = " << double(t) << "." << endl;
}