
#include "LaskentaLite.h"

#include <iomanip>
#include <vector>

using std::cout;
using std::endl;

//**********************************************************************************************************************

Expression ConstructFunction(Expression const &x, int n)
{
    switch (n)
    {
    case 0:return abs(x);
    case 1:return sgn(x);
    case 2:return sqrt(x);
    case 3:return cbrt(x);
    case 4:return exp(x);
    case 5:return expm1(x);
    case 6:return log(x);
    case 7:return log1p(x);
    case 8:return sin(x);
    case 9:return cos(x);
    case 10:return tan(x);
    case 11:return asin(x);
    case 12:return acos(x);
    case 13:return atan(x);
    case 14:return sinh(x);
    case 15:return cosh(x);
    case 16:return tanh(x);
    case 17:return asinh(x);
    case 18:return acosh(x);
    case 19:return atanh(x);
    case 20:return erf(x);
    case 21:return erfc(x);
    case 22:return 1 / x;
    case 23:return x < 0;
    case 24:return !x;
    case 25:return -x;
    case 26:return x * x;
    case 27:return ceil(x);
    case 28:return floor(x);
    case 29:return sqrt(x) / x;
    case 30:return x / sqrt(x);
    }

    return 0;
}

static int N_max = 31;

//**********************************************************************************************************************

double SimilarityPercentage(double a, double b)
{
    if (isnan(a) || isnan(b)) return -1;
    if (sgn(a) != sgn(b)) return 0;
    if (abs(a) < abs(b)) return 100 * a / b;
    if (abs(a) > abs(b)) return 100 * b / a;
    return 100;
}

//**********************************************************************************************************************

double FiniteDifferenceTest(Expression const &f, Variable &x, double d, double epsilon)
{
    x = d; double dx = double(f.Derive(x));
    x = d - epsilon; double y0 = double(f);
    x = d + epsilon; double y1 = double(f);
    return SimilarityPercentage(dx, (y1 - y0) / (2 * epsilon));
}

//**********************************************************************************************************************

double HigherDerivativeTest(Expression const &f, Variable &x, double d)
{
    Expression a = (1 / (2 * f * f + 1)).Derive(x, 6)[6];
    Expression b = (1 / (2 * f * f + 1)).Derive(x).Derive(x).Derive(x).Derive(x).Derive(x).Derive(x);
    x = d;
    return SimilarityPercentage(double(a), double(b));
}

//**********************************************************************************************************************

void ReportTest(Expression const &f, Variable &x)
{
    double t;
    t = FiniteDifferenceTest(f, x, -0.4, 1e-4);
    if (t < 0) t = FiniteDifferenceTest(f, x, 1.1, 1e-4);
    if (t < 0) t = FiniteDifferenceTest(f, x, 0.4, 1e-4);
    cout << t << "%\t";
    t = HigherDerivativeTest(f, x, -0.4);
    if (t < 0) t = HigherDerivativeTest(f, x, 1.1);
    if (t < 0) t = HigherDerivativeTest(f, x, 0.4);
    cout << t << "%\t";
    cout << f << endl;
}

//**********************************************************************************************************************

bool test()
{
    cout << std::setprecision(4);

    Variable x; x.Name("x");
    Variable y; y.Name("y");
    Expression f;

#if defined(_DEBUG)
    cout << endl << "Debug build" << endl << endl;
#else
    cout << endl << "Release build" << endl << endl;
#endif
    cout << "1) Finite difference test" << endl;
    cout << "2) Higher derivative test" << endl;
    cout << endl << "1)\t2)\tFunction:" << endl;

    for (int i = 0; i < N_max; ++i)
    {
        ReportTest(ConstructFunction(x, i), x);
    }

    ReportTest(pow(2, x), x);
    ReportTest(pow(x, 3), x);
    ReportTest(pow(x, 5.5), x);
    cout << "- - -" << endl;
    ReportTest((2 * x) + (3 * x) + (4 * y), x);
    ReportTest((2 + x) * (3 + x) * (4 + y), x);
    ReportTest(pow(x, 1 / x), x);
    ReportTest(atan2(x, 1 / x), x);
    ReportTest(fmod(x, 1 / x), x);
    ReportTest(hypot(x, 1 / x), x);
    ReportTest(fma(x + 1, x + 2, 1 / x), x);
    ReportTest(cond(x > 0 && x < 1, 1 / x, x *x), x);
    cout << "- - -" << endl;
    ReportTest(erf(1 / expm1(2 * sin(pow(log1p(x), 3) + 1))), x);
    ReportTest(pow(atan2(x, x * x + 1), hypot(x, x * x + 1)), x);

    return true;
}

//**********************************************************************************************************************
