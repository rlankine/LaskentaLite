
/*
MIT License

Copyright (c) 2025 Risto Lankinen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <assert.h>
#include <iostream>
#include <string>
#include <unordered_map>

using std::cout;
using std::endl;

struct Expression;

//**********************************************************************************************************************

// #define VERBOSE

#if defined(_DEBUG)
#define noexcept
#define UNREACHABLE do { std::cerr << std::endl << "Reached code deemed unreachable in '" << __FUNCTION__ << "(...)' line " << __LINE__ << "." << std::endl; throw nullptr; } while(false)
#else
#define UNREACHABLE do { std::terminate(); } while(false)
#endif

/***********************************************************************************************************************
*** Shared
***********************************************************************************************************************/

struct Shared
{
	template <typename T, typename = typename std::enable_if<std::is_base_of<Shared, T>::value>::type>
	[[nodiscard]] static inline T const *Clone(T const *p) noexcept { assert(p); ++p->nShared; return p; }

	template <typename T, typename = typename std::enable_if<std::is_base_of<Shared, T>::value>::type>
	[[nodiscard]] static inline T const *Clone(T const &r) noexcept { ++r.nShared; return &r; }

	template <typename T, typename = typename std::enable_if<std::is_base_of<Shared, T>::value>::type>
	[[nodiscard]] static inline T *Clone(T *p) noexcept { assert(p); ++p->nShared; return p; }

	template <typename T, typename = typename std::enable_if<std::is_base_of<Shared, T>::value>::type>
	[[nodiscard]] static inline T *Clone(T &r) noexcept { ++r.nShared; return &r; }

	static inline void Erase(Shared const *p) noexcept { assert(p); if (!--p->nShared) delete p; }

protected:
	Shared() noexcept : nShared(1) { }
	virtual ~Shared() noexcept = 0 { assert(nShared < 2); }

private:
	mutable size_t nShared;

	Shared(Shared &&) = delete;
	Shared(Shared const &) = delete;
	Shared &operator=(Shared &&) = delete;
	Shared &operator=(Shared const &) = delete;
};

/***********************************************************************************************************************
*** Literals
***********************************************************************************************************************/

namespace Literal
{
static struct Epsilon final { } const eps;
double const inf = std::numeric_limits<double>::infinity();
double const nan = std::numeric_limits<double>::quiet_NaN();
}

inline double operator+(double d, Literal::Epsilon) noexcept { return nextafter(d, Literal::inf); }
inline double operator-(double d, Literal::Epsilon) noexcept { return nextafter(d, -Literal::inf); }
inline double &operator+=(double &d, Literal::Epsilon) noexcept { return d = nextafter(d, Literal::inf); }
inline double &operator-=(double &d, Literal::Epsilon) noexcept { return d = nextafter(d, -Literal::inf); }

/***********************************************************************************************************************
*** Bounds
***********************************************************************************************************************/

struct Bounds final
{
	Bounds(double a, double b) noexcept : lower(isnan(b) ? b : a), upper(isnan(a) ? a : b) { assert(isnan(lower) == isnan(upper)); }

	bool operator<(double d) const noexcept { return upper < d && !bipartite(); }
	bool operator<=(double d) const noexcept { return upper <= d && !bipartite(); }
	bool operator>=(double d) const noexcept { return lower >= d && !bipartite(); }
	bool operator>(double d) const noexcept { return lower > d && !bipartite(); }

	Bounds above(double d) const noexcept
	{
		if (isnan(d)) return *this;
		return {excludes(d) ? lower : d, bipartite() ? Literal::inf : upper < d ? Literal::nan : upper};
	}

	Bounds below(double d) const noexcept
	{
		if (isnan(d)) return *this;
		return {bipartite() ? -Literal::inf : d < lower ? Literal::nan : lower, excludes(d) ? upper : d};
	}

	bool bipartite() const noexcept { return (upper < lower); }
	bool excludes(double d) const noexcept { return ((upper < d) != (d < lower) != bipartite()) || isnan(lower) || isnan(d); }
	bool includes(double d) const noexcept { return !excludes(d); }

	friend std::ostream &operator<<(std::ostream &out, Bounds const &r)
	{
		if (r.bipartite()) return out << "[-inf.." << r.upper << "]..[" << r.lower << "..inf]";
		return out << "[" << r.lower << ".." << r.upper << "]";
	}

	double const lower;
	double const upper;
};

/***********************************************************************************************************************
*** Variable
***********************************************************************************************************************/

struct Variable final
{
	struct data : public Shared
	{
		data(double d, Bounds const &r) : value(r.lower), bounds(r), name("[&" + std::to_string(size_t(this)) + "]") { assign(d); }

		void assign(double d) noexcept
		{
			if (isnan(d)) return;
			if (bounds.excludes(d)) d = value < bounds.lower == d < bounds.lower == bounds.bipartite() ? bounds.lower : bounds.upper;
			if (value == d) return;
			value = d;
			Touch();
		}

		double evaluate() const noexcept { return value; }

	private:
		Bounds const bounds;
		double value;

	public:
		std::string name;
	};

	Variable(double = 0, Bounds const &_ = {-Literal::inf, Literal::inf});
	Variable(Variable const &) noexcept;
	explicit Variable(Expression const &);

	~Variable() noexcept;

	Variable &operator=(Variable const &) noexcept;
	Variable &operator=(double) noexcept;
	Variable &operator+=(Literal::Epsilon) noexcept { return operator=(nextafter(operator double(), Literal::inf)); }
	Variable &operator-=(Literal::Epsilon) noexcept { return operator=(nextafter(operator double(), -Literal::inf)); }
	Variable &operator+=(double d) noexcept { return operator=(operator double() + d); }
	Variable &operator-=(double d) noexcept { return operator=(operator double() - d); }
	Variable &operator*=(double d) noexcept { return operator=(operator double() * d); }
	Variable &operator/=(double d) noexcept { return operator=(operator double() / d); }
	data const *operator->() const noexcept { return pData; }
	explicit operator double() const noexcept { return pData->evaluate(); }
	size_t id() const noexcept { return size_t(pData); }
	bool is(Variable const &) const noexcept;
	std::string Name() const;
	Variable &Name(std::string const &);
	static bool Clean(size_t &) noexcept;
	static void Touch() noexcept;

private:
	data *pData;

	friend struct VariableNode;
};

/***********************************************************************************************************************
*** Bindings
***********************************************************************************************************************/

struct Bindings final
{
	Bindings() = default;
	Bindings(Bindings const &);

	~Bindings();

	Bindings &operator=(Bindings const &);

	Bindings &Insert(Variable const &, Expression const &);
	Bindings &Remove(Variable const &);
	void Commit() noexcept;
	void Rollback() noexcept;
	void Snapshot() noexcept;

private:
	struct item;
	std::unordered_map<size_t, item *> bindings;

	friend struct VariableNode;
};

/***********************************************************************************************************************
*** Expression
***********************************************************************************************************************/

struct Expression final
{
	struct data : public Shared
	{
		enum class NodeType
		{
			NOT_A_NUMBER, CONSTANT, VARIABLE, ABS, SGN, SQRT, CBRT, EXP, EXPM1, LOG, LOG1P, SIN, COS, TAN,
			ASIN, ACOS, ATAN, SINH, COSH, TANH, ASINH, ACOSH, ATANH, ERF, ERFC, CEIL, FLOOR,
			INVERT, NEGATE, SQUARE, ADD, MUL, POW, ATAN2, FMOD, EQUAL, HYPOT, LESS, NEXTAFTER, COND, FMA
		};

		[[nodiscard]] static data const *create(double);
		[[nodiscard]] static data const *create(Variable const &);
		[[nodiscard]] virtual data const *create(NodeType) const;
		[[nodiscard]] virtual data const *create(NodeType, Expression const &) const;
		[[nodiscard]] virtual data const *create(NodeType, Expression const &, Expression const &) const;

		virtual Expression this_() const noexcept final { return Clone(this); }

		// Functions

		virtual Expression abs_() const { return create(NodeType::ABS); }
		virtual Expression sgn_() const { return create(NodeType::SGN); }
		virtual Expression sqrt_() const { return create(NodeType::SQRT); }
		virtual Expression cbrt_() const { return create(NodeType::CBRT); }
		virtual Expression exp_() const { return create(NodeType::EXP); }
		virtual Expression expm1_() const { return create(NodeType::EXPM1); }
		virtual Expression log_() const { return create(NodeType::LOG); }
		virtual Expression log1p_() const { return create(NodeType::LOG1P); }
		virtual Expression sin_() const { return create(NodeType::SIN); }
		virtual Expression cos_() const { return create(NodeType::COS); }
		virtual Expression tan_() const { return create(NodeType::TAN); }
		virtual Expression asin_() const { return create(NodeType::ASIN); }
		virtual Expression acos_() const { return create(NodeType::ACOS); }
		virtual Expression atan_() const { return create(NodeType::ATAN); }
		virtual Expression sinh_() const { return create(NodeType::SINH); }
		virtual Expression cosh_() const { return create(NodeType::COSH); }
		virtual Expression tanh_() const { return create(NodeType::TANH); }
		virtual Expression asinh_() const { return create(NodeType::ASINH); }
		virtual Expression acosh_() const { return create(NodeType::ACOSH); }
		virtual Expression atanh_() const { return create(NodeType::ATANH); }
		virtual Expression erf_() const { return create(NodeType::ERF); }
		virtual Expression erfc_() const { return create(NodeType::ERFC); }
		virtual Expression ceil_() const { return create(NodeType::CEIL); }
		virtual Expression floor_() const { return create(NodeType::FLOOR); }
		virtual Expression invert_() const { return create(NodeType::INVERT); }
		virtual Expression negate_() const { return create(NodeType::NEGATE); }
		virtual Expression square_() const { return create(NodeType::SQUARE); }

		// Operators

		virtual Expression add_(Expression const &r) const;
		virtual Expression mul_(Expression const &r) const;
		virtual Expression pow_(Expression const &r) const;
		virtual Expression atan2_(Expression const &r) const { return create(NodeType::ATAN2, r); }
		virtual Expression equal_(Expression const &r) const { return create(NodeType::EQUAL, r); }
		virtual Expression fmod_(Expression const &r) const { return create(NodeType::FMOD, r); }
		virtual Expression hypot_(Expression const &r) const;
		virtual Expression less_(Expression const &r) const { return create(NodeType::LESS, r); }
		virtual Expression nextafter_(Expression const &r) const { return create(NodeType::NEXTAFTER, r); }

		// Ternaries

		virtual Expression cond_(Expression const &r, Expression const &s) const { return create(NodeType::COND, r, s); }
		virtual Expression fma_(Expression const &r, Expression const &s) const { return create(NodeType::FMA, r, s); }

		// Evaluation and derivation

		[[nodiscard]] virtual Expression bind(Bindings const &) const = 0;
		[[nodiscard]] virtual Expression derive(Variable const &) const = 0;
		[[nodiscard]] virtual double evaluate() const noexcept { if (Variable::Clean(state)) return value; return value = compute(); }

		// Optimization tools

		bool is(data const *p) const noexcept { return this == p; }
		bool is(NodeType t) const noexcept { return nodeType == t; }

		// Cache management

		virtual void flush() const noexcept { if (cache) { Erase(cache); cache = nullptr; } }

		// Printing

		enum class Context { DEFAULT, INVERT, BOOL, NOT, NEGATE, ADD, MUL };

		virtual void print(std::ostream &, Context = Context::DEFAULT) const = 0;

	protected:
		data(NodeType t) : nodeType(t), state(0), value(0), cache(nullptr) { }
		virtual ~data() { }
		NodeType const nodeType;
		mutable size_t state;
		mutable double value;
		mutable data const *cache;

	private:
		virtual double compute() const noexcept = 0;
		void *operator new(size_t n) { return ::operator new(n); }
		void *operator new[](size_t) = delete;
	};

	Expression();
	Expression(Expression const &) noexcept;
	Expression(double);
	Expression(Variable const &);
	Expression(data const *) noexcept;
	template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type> Expression(T n) : Expression(double(n)) { }
	~Expression();

	Expression &operator=(Expression const &) noexcept;

	[[nodiscard]] explicit operator double() const noexcept { return pData->evaluate(); }
	[[nodiscard]] Expression Bind(Bindings const &) const;
	[[nodiscard]] Expression Derive(Variable const &) const;
	[[nodiscard]] std::vector<Expression> Derive(Variable const &, int) const;

	mutable data const *pData = nullptr;
	data const *operator->() const noexcept { return pData; }

	friend std::ostream &operator<<(std::ostream &, Expression const &);
};

/***********************************************************************************************************************
*** Functions
***********************************************************************************************************************/

// F(Expression) -> Expression

inline Expression abs(Expression const &r) { return r->abs_(); }
inline Expression sgn(Expression const &r) { return r->sgn_(); }
inline Expression sqrt(Expression const &r) { return r->sqrt_(); }
inline Expression cbrt(Expression const &r) { return r->cbrt_(); }
inline Expression exp(Expression const &r) { return r->exp_(); }
inline Expression expm1(Expression const &r) { return r->expm1_(); }
inline Expression log(Expression const &r) { return r->log_(); }
inline Expression log1p(Expression const &r) { return r->log1p_(); }
inline Expression sin(Expression const &r) { return r->sin_(); }
inline Expression cos(Expression const &r) { return r->cos_(); }
inline Expression tan(Expression const &r) { return r->tan_(); }
inline Expression asin(Expression const &r) { return r->asin_(); }
inline Expression acos(Expression const &r) { return r->acos_(); }
inline Expression atan(Expression const &r) { return r->atan_(); }
inline Expression sinh(Expression const &r) { return r->sinh_(); }
inline Expression cosh(Expression const &r) { return r->cosh_(); }
inline Expression tanh(Expression const &r) { return r->tanh_(); }
inline Expression asinh(Expression const &r) { return r->asinh_(); }
inline Expression acosh(Expression const &r) { return r->acosh_(); }
inline Expression atanh(Expression const &r) { return r->atanh_(); }
inline Expression erf(Expression const &r) { return r->erf_(); }
inline Expression erfc(Expression const &r) { return r->erfc_(); }
inline Expression ceil(Expression const &r) { return r->ceil_(); }
inline Expression floor(Expression const &r) { return r->floor_(); }

inline Expression operator+(Expression const &r, Literal::Epsilon) { return r->nextafter_(Literal::inf); }
inline Expression operator-(Expression const &r, Literal::Epsilon) { return r->nextafter_(-Literal::inf); }
inline Expression operator+(Expression const &r) noexcept { return r; }
inline Expression operator-(Expression const &r) { return r->negate_(); }
inline Expression operator!(Expression const &r) { return r->equal_(0); }

// F(Expression, Expression) -> Expression

inline Expression pow(Expression const &r, Expression const &s) { return r->pow_(s); }
inline Expression atan2(Expression const &r, Expression const &s) { return r->atan2_(s); }
inline Expression fmod(Expression const &r, Expression const &s) { return r->fmod_(s); }
inline Expression hypot(Expression const &r, Expression const &s) { return r->hypot_(s); }
inline Expression nextafter(Expression const &r, Expression const &s) { return r->nextafter_(s); }
inline Expression max(Expression const &r, Expression const &s) { return r->less_(s)->cond_(s, r); }
inline Expression min(Expression const &r, Expression const &s) { return r->less_(s)->cond_(r, s); }

inline Expression operator+(Expression const &r, Expression const &s) { return r->add_(s); }
inline Expression operator-(Expression const &r, Expression const &s) { return r->add_(s->negate_()); }
inline Expression operator*(Expression const &r, Expression const &s) { return r->mul_(s); }
inline Expression operator/(Expression const &r, Expression const &s) { return r->mul_(s->invert_()); }
inline Expression operator<(Expression const &r, Expression const &s) { return r->less_(s); }
inline Expression operator==(Expression const &r, Expression const &s) { return r->equal_(s); }

inline Expression &operator+=(Expression &r, Literal::Epsilon) { return r = nextafter(r, Literal::inf); }
inline Expression &operator-=(Expression &r, Literal::Epsilon) { return r = nextafter(r, -Literal::inf); }
inline Expression &operator+=(Expression &r, Expression const &s) { return r = r + s; }
inline Expression &operator-=(Expression &r, Expression const &s) { return r = r - s; }
inline Expression &operator*=(Expression &r, Expression const &s) { return r = r * s; }
inline Expression &operator/=(Expression &r, Expression const &s) { return r = r / s; }
inline Expression operator>(Expression const &r, Expression const &s) { return s < r; }
inline Expression operator<=(Expression const &r, Expression const &s) { return !(s < r); }
inline Expression operator>=(Expression const &r, Expression const &s) { return !(r < s); }
inline Expression operator!=(Expression const &r, Expression const &s) { return !(r == s); }
inline Expression operator&&(Expression const &r, Expression const &s) { return (!r) < (!!s); }
inline Expression operator||(Expression const &r, Expression const &s) { return !((!!r) < (!s)); }

// F(Expression, Expression, Expression) -> Expression

inline Expression cond(Expression const &r, Expression const &s, Expression const &t) { return r->cond_(s, t); }
inline Expression fma(Expression const &r, Expression const &s, Expression const &t) { return r->fma_(s, t); }

// F(Variable) -> Expression

inline Expression abs(Variable const &r) { return abs(Expression(r)); }
inline Expression sgn(Variable const &r) { return sgn(Expression(r)); }
inline Expression sqrt(Variable const &r) { return sqrt(Expression(r)); }
inline Expression cbrt(Variable const &r) { return cbrt(Expression(r)); }
inline Expression exp(Variable const &r) { return exp(Expression(r)); }
inline Expression expm1(Variable const &r) { return expm1(Expression(r)); }
inline Expression log(Variable const &r) { return log(Expression(r)); }
inline Expression log1p(Variable const &r) { return log1p(Expression(r)); }
inline Expression sin(Variable const &r) { return sin(Expression(r)); }
inline Expression cos(Variable const &r) { return cos(Expression(r)); }
inline Expression tan(Variable const &r) { return tan(Expression(r)); }
inline Expression asin(Variable const &r) { return asin(Expression(r)); }
inline Expression acos(Variable const &r) { return acos(Expression(r)); }
inline Expression atan(Variable const &r) { return atan(Expression(r)); }
inline Expression sinh(Variable const &r) { return sinh(Expression(r)); }
inline Expression cosh(Variable const &r) { return cosh(Expression(r)); }
inline Expression tanh(Variable const &r) { return tanh(Expression(r)); }
inline Expression asinh(Variable const &r) { return asinh(Expression(r)); }
inline Expression acosh(Variable const &r) { return acosh(Expression(r)); }
inline Expression atanh(Variable const &r) { return atanh(Expression(r)); }
inline Expression erf(Variable const &r) { return erf(Expression(r)); }
inline Expression erfc(Variable const &r) { return erfc(Expression(r)); }
inline Expression ceil(Variable const &r) { return ceil(Expression(r)); }
inline Expression floor(Variable const &r) { return floor(Expression(r)); }

inline Expression operator+(Variable const &r, Literal::Epsilon) { return Expression(r) + Literal::eps; }
inline Expression operator-(Variable const &r, Literal::Epsilon) { return Expression(r) - Literal::eps; }
inline Expression operator+(Variable const &r) { return +Expression(r); }
inline Expression operator-(Variable const &r) { return -Expression(r); }
inline Expression operator!(Variable const &r) { return !Expression(r); }

// F(Variable, Variable) -> Expression

inline Expression pow(Variable const &r, Variable const &s) { return pow(Expression(r), s); }
inline Expression atan2(Variable const &r, Variable const &s) { return atan2(Expression(r), s); }
inline Expression fmod(Variable const &r, Variable const &s) { return fmod(Expression(r), s); }
inline Expression hypot(Variable const &r, Variable const &s) { return hypot(Expression(r), s); }
inline Expression max(Variable const &r, Variable const &s) { return max(Expression(r), s); }
inline Expression min(Variable const &r, Variable const &s) { return min(Expression(r), s); }
inline Expression nextafter(Variable const &r, Variable const &s) { return nextafter(Expression(r), s); }

inline Expression operator+(Variable const &r, Variable const &s) { return Expression(r) + s; }
inline Expression operator-(Variable const &r, Variable const &s) { return Expression(r) - s; }
inline Expression operator*(Variable const &r, Variable const &s) { return Expression(r) * s; }
inline Expression operator/(Variable const &r, Variable const &s) { return Expression(r) / s; }
inline Expression operator<(Variable const &r, Variable const &s) { return Expression(r) < s; }
inline Expression operator>(Variable const &r, Variable const &s) { return Expression(r) > s; }
inline Expression operator<=(Variable const &r, Variable const &s) { return Expression(r) <= s; }
inline Expression operator>=(Variable const &r, Variable const &s) { return Expression(r) >= s; }
inline Expression operator!=(Variable const &r, Variable const &s) { return Expression(r) != s; }
inline Expression operator==(Variable const &r, Variable const &s) { return Expression(r) == s; }
inline Expression operator&&(Variable const &r, Variable const &s) { return Expression(r) && s; }
inline Expression operator||(Variable const &r, Variable const &s) { return Expression(r) || s; }

// F(Variable, double) -> Expression

inline Expression pow(Variable const &r, double s) { return pow(Expression(r), s); }
inline Expression atan2(Variable const &r, double s) { return atan2(Expression(r), s); }
inline Expression fmod(Variable const &r, double s) { return fmod(Expression(r), s); }
inline Expression hypot(Variable const &r, double s) { return hypot(Expression(r), s); }
inline Expression max(Variable const &r, double s) { return max(Expression(r), s); }
inline Expression min(Variable const &r, double s) { return min(Expression(r), s); }
inline Expression nextafter(Variable const &r, double s) { return nextafter(Expression(r), s); }

inline Expression operator+(Variable const &r, double s) { return Expression(r) + s; }
inline Expression operator-(Variable const &r, double s) { return Expression(r) - s; }
inline Expression operator*(Variable const &r, double s) { return Expression(r) * s; }
inline Expression operator/(Variable const &r, double s) { return Expression(r) / s; }
inline Expression operator<(Variable const &r, double s) { return Expression(r) < s; }
inline Expression operator>(Variable const &r, double s) { return Expression(r) > s; }
inline Expression operator<=(Variable const &r, double s) { return Expression(r) <= s; }
inline Expression operator>=(Variable const &r, double s) { return Expression(r) >= s; }
inline Expression operator!=(Variable const &r, double s) { return Expression(r) != s; }
inline Expression operator==(Variable const &r, double s) { return Expression(r) == s; }
inline Expression operator&&(Variable const &r, double s) { return Expression(r) && s; }
inline Expression operator||(Variable const &r, double s) { return Expression(r) || s; }

// F(double, Variable) -> Expression

inline Expression pow(double r, Variable const &s) { return pow(Expression(r), s); }
inline Expression atan2(double r, Variable const &s) { return atan2(Expression(r), s); }
inline Expression fmod(double r, Variable const &s) { return fmod(Expression(r), s); }
inline Expression hypot(double r, Variable const &s) { return hypot(Expression(r), s); }
inline Expression max(double r, Variable const &s) { return max(Expression(r), s); }
inline Expression min(double r, Variable const &s) { return min(Expression(r), s); }
inline Expression nextafter(double r, Variable const &s) { return min(Expression(r), s); }

inline Expression operator+(double r, Variable const &s) { return Expression(r) + s; }
inline Expression operator-(double r, Variable const &s) { return Expression(r) - s; }
inline Expression operator*(double r, Variable const &s) { return Expression(r) * s; }
inline Expression operator/(double r, Variable const &s) { return Expression(r) / s; }
inline Expression operator<(double r, Variable const &s) { return Expression(r) < s; }
inline Expression operator>(double r, Variable const &s) { return Expression(r) > s; }
inline Expression operator<=(double r, Variable const &s) { return Expression(r) <= s; }
inline Expression operator>=(double r, Variable const &s) { return Expression(r) >= s; }
inline Expression operator!=(double r, Variable const &s) { return Expression(r) != s; }
inline Expression operator==(double r, Variable const &s) { return Expression(r) == s; }
inline Expression operator&&(double r, Variable const &s) { return Expression(r) && s; }
inline Expression operator||(double r, Variable const &s) { return Expression(r) || s; }

// F(Variable, Variable, Variable) -> Expression

inline Expression cond(Variable const &r, Variable const &s, Variable const &t) { return cond(Expression(r), s, t); }
inline Expression fma(Variable const &r, Variable const &s, Variable const &t) { return fma(Expression(r), s, t); }

// F(Variable, Variable, double) -> Expression

inline Expression cond(Variable const &r, Variable const &s, double t) { return cond(Expression(r), s, t); }
inline Expression fma(Variable const &r, Variable const &s, double t) { return fma(Expression(r), s, t); }

// F(Variable, double, Variable) -> Expression

inline Expression cond(Variable const &r, double s, Variable const &t) { return cond(Expression(r), s, t); }
inline Expression fma(Variable const &r, double s, Variable const &t) { return fma(Expression(r), s, t); }

// F(Variable, double, double) -> Expression

inline Expression cond(Variable const &r, double s, double t) { return cond(Expression(r), s, t); }
inline Expression fma(Variable const &r, double s, double t) { return fma(Expression(r), s, t); }

// F(double, Variable, Variable) -> Expression

inline Expression cond(double r, Variable const &s, Variable const &t) { return r == 0 ? Expression(t) : Expression(s); }
inline Expression fma(double r, Variable const &s, Variable const &t) { return fma(Expression(r), s, t); }

// F(double, Variable, double) -> Expression

inline Expression cond(double r, Variable const &s, double t) { return r == 0 ? Expression(t) : Expression(s); }
inline Expression fma(double r, Variable const &s, double t) { return fma(Expression(r), s, t); }

// F(double, double, Variable) -> Expression

inline Expression cond(double r, double s, Variable const &t) { return r == 0 ? Expression(t) : Expression(s); }
inline Expression fma(double r, double s, Variable const &t) { return fma(Expression(r), s, t); }

// F(double) -> double

inline double sgn(double d) noexcept { return isnan(d) ? d : double(d > 0) - (d < 0); }         // Sign

// F(double, double, double) -> double

inline double cond(double a, double b, double c) noexcept { return a == 0 ? c : b; }            // Function version of non-overloadable C++ conditional operator

// ---------------------------------------------------------------------------------------------------------------------

inline std::ostream &operator<<(std::ostream &out, Expression const &r)
{
	r->print(out);
	return out;
}

/***********************************************************************************************************************
*** Variable
***********************************************************************************************************************/

namespace
{
inline static size_t valid(bool b = false)
{
	static size_t result = 1ULL;
	if (b) ++result;
	return result;
}
}

inline Variable::Variable(double d, Bounds const &r) : pData(new data(d, r))
{
}

inline Variable::Variable(Variable const &r) noexcept : pData(Shared::Clone(r.pData))
{
}

inline Variable::Variable(Expression const &r) : Variable(double(r))
{
}

inline Variable::~Variable() noexcept
{
	Shared::Erase(pData);
}

inline Variable &Variable::operator=(Variable const &r) noexcept
{
	pData->assign(double(r));
	return *this;
}

inline Variable &Variable::operator=(double d) noexcept
{
	pData->assign(d);
	return *this;
}

inline bool Variable::is(Variable const &r) const noexcept
{
	return pData == r.pData;
}

inline std::string Variable::Name() const
{
	return pData->name;
}

inline Variable &Variable::Name(std::string const &r)
{
	pData->name = r;
	return *this;
}

inline bool Variable::Clean(size_t &r) noexcept
{
	return r != valid() ? r = valid(), false : true;
}

inline void Variable::Touch() noexcept
{
	valid(true);
}

/***********************************************************************************************************************
*** Bindings
***********************************************************************************************************************/

struct Bindings::item final
{
	Expression const function;
	mutable Variable variable;
	mutable double d;

	void commit() noexcept { d = double(function); }
	void rollback() noexcept { variable = d; }
	void snapshot() noexcept { d = double(variable); }
};

inline Bindings::Bindings(Bindings const &r)
{
	for (auto const &node : r.bindings)
	{
		bindings[node.first] = new item({node.second->function, node.second->variable, node.second->variable->evaluate()});
	}
}

inline Bindings::~Bindings()
{
	for (auto const &node : bindings) delete node.second;
}

inline Bindings &Bindings::operator=(Bindings const &r)
{
	if (this != &r)
	{
		this->~Bindings();
		new(this) Bindings(r);
	}
}

inline Bindings &Bindings::Insert(Variable const &r, Expression const &s)
{
	if (bindings.find(r.id()) != bindings.end()) Remove(r);
	bindings[r.id()] = new item({s, r, r->evaluate()});
	return *this;
}

inline Bindings &Bindings::Remove(Variable const &r)
{
	auto const node = bindings.find(r.id());
	if (node != bindings.end())
	{
		delete node->second;
		bindings.erase(node);
	}
	return *this;
}

inline void Bindings::Commit() noexcept
{
	for (auto const &node : bindings) node.second->commit();
	Rollback();
}

inline void Bindings::Rollback() noexcept
{
	for (auto const &node : bindings) node.second->rollback();
}

inline void Bindings::Snapshot() noexcept
{
	for (auto const &node : bindings) node.second->snapshot();
}

/***********************************************************************************************************************
*** ObjectGuard
***********************************************************************************************************************/

#if !defined(_DEBUG) && !defined(VERBOSE)
template <typename = void> struct ObjectGuard
{
};
#else

#pragma intrinsic(memcpy)

namespace
{
struct Guarded
{
	mutable Guarded const *pNext;
	mutable Guarded const *pPrev;
	size_t nSerialNumber;

	static size_t nCreationCount;

	Guarded() noexcept : pNext(this), pPrev(this), nSerialNumber(++nCreationCount)
	{
	}

	Guarded(Guarded const &r) noexcept : nSerialNumber(++nCreationCount), pNext(&r), pPrev(pNext->pPrev)
	{
		pNext->pPrev = this;
		pPrev->pNext = this;
	}

	virtual ~Guarded() noexcept
	{
		pNext->pPrev = pPrev;
		pPrev->pNext = pNext;
	}

	Guarded &operator=(Guarded const &) = delete;
};

size_t Guarded::nCreationCount = 0;
} // namespace

template <typename> struct ObjectGuard : public Guarded
{
	static struct data final
	{
		Guarded rootNode;
		char const *szClassName;
		unsigned int nCreated;
		unsigned int nDeleted;
		unsigned int nHighest;

		data() : rootNode(), szClassName(nullptr), nCreated(0), nDeleted(0), nHighest(0)
		{
			size_t const PREFIX = sizeof "ObjectGuard<" - 1;
			size_t const SUFFIX = sizeof ">::data::data" - 1;
			size_t const size = sizeof __FUNCTION__ - PREFIX - SUFFIX;
			static char buffer[size];

			memcpy(buffer, __FUNCTION__ + PREFIX, size);
			buffer[size - 1] = '\0';
			szClassName = buffer;
		}

		~data()
		{
			int limit = 8;
			switch (nCreated - nDeleted)
			{
			case 0:
#if defined(VERBOSE)
				std::cerr << "Created and deleted " << nCreated << " objects of type <" << szClassName << "> (of which at most " << nHighest << " existed simultaneously)" << std::endl;
#endif
				break;

			case 1:
				std::cerr << "Object Guard: 1 object of type <" << szClassName << "> was leaked." << std::endl
					<< "   Serial number of the leaked object is " << rootNode.pNext->nSerialNumber << ")." << std::endl
					<< std::endl;
				break;

			default:
				std::cerr << "Object Guard: " << nCreated - nDeleted << " objects of type <" << szClassName << "> were leaked." << std::endl
					<< "   Serial numbers of the leaked objects are ";
				for (Guarded const *p = rootNode.pNext; p != &rootNode; p = p->pNext)
				{
					if (--limit < 0)
					{
						std::cerr << "... and more";
						break;
					}
					if (p->pNext == &rootNode)
					{
						std::cerr << " and ";
						++limit;
					}
					else if (p != rootNode.pNext)
					{
						std::cerr << ", ";
					}
					std::cerr << p->nSerialNumber;
				}
				std::cerr << "." << std::endl
					<< std::endl;
				break;
			}
		}

		data &operator++()
		{
			if (++nCreated > nHighest + nDeleted) { ++nHighest; }
			return *this;
		}

		data &operator--()
		{
			++nDeleted;
			return *this;
		}

	} &instance()
	{
		static data d;
		return d;
	};

	ObjectGuard() : Guarded(instance().rootNode) { ++instance(); }
	ObjectGuard(ObjectGuard const &) : Guarded(instance().rootNode) { ++instance(); }

protected:
	~ObjectGuard() { --instance(); }
};

#endif

/***********************************************************************************************************************
*** Key
***********************************************************************************************************************/

struct Key final
{
	Key(double d) : nodeType(Expression::data::NodeType::CONSTANT)
	{
		assert(sizeof data >= sizeof(double));

		memset(&data, 0, sizeof data);
		data.value = d;
	}

	Key(size_t s) : nodeType(Expression::data::NodeType::VARIABLE)
	{
		assert(sizeof data >= sizeof(size_t));

		memset(&data, 0, sizeof data);
		data.id = s;
	}

	Key(Expression::data::NodeType n, Expression::data const *p1) : nodeType(n)
	{
		assert(sizeof data >= sizeof(Expression::data const *));

		memset(&data, 0, sizeof data);
		data.param1 = p1;
	}

	Key(Expression::data::NodeType n, Expression::data const *p1, Expression::data const *p2) : nodeType(n)
	{
		assert(sizeof data >= sizeof(Expression::data const *) * 2);

		memset(&data, 0, sizeof data);
		data.param1 = p1;
		data.param2 = p2;
	}

	Key(Expression::data::NodeType n, Expression::data const *p1, Expression::data const *p2, Expression::data const *p3) : nodeType(n)
	{
		assert(sizeof data >= sizeof(Expression::data const *) * 3);

		memset(&data, 0, sizeof data);
		data.param1 = p1;
		data.param2 = p2;
		data.param3 = p3;
	}

	Expression::data::NodeType const nodeType;

	union
	{
		double value;
		size_t id;

		struct
		{
			Expression::data const *param1;
			Expression::data const *param2;
			Expression::data const *param3;
		};
	} data;

	bool operator==(Key const &r) const
	{
		if (nodeType != r.nodeType) return false;

		switch (nodeType)
		{
		case Expression::data::NodeType::CONSTANT:
			return data.value == r.data.value;

		case Expression::data::NodeType::VARIABLE:
			return data.id == r.data.id;

		case Expression::data::NodeType::ADD:
		case Expression::data::NodeType::MUL:
		case Expression::data::NodeType::EQUAL:
		case Expression::data::NodeType::HYPOT:
			if (data.param1 == r.data.param1 && data.param2 == r.data.param2) return true;
			if (data.param1 == r.data.param2 && data.param2 == r.data.param1) return true;
			return false;

		case Expression::data::NodeType::POW:
		case Expression::data::NodeType::ATAN2:
		case Expression::data::NodeType::FMOD:
		case Expression::data::NodeType::LESS:
		case Expression::data::NodeType::NEXTAFTER:
			return data.param1 == r.data.param1 && data.param2 == r.data.param2;

		case Expression::data::NodeType::FMA:
			if (data.param1 == r.data.param1 && data.param2 == r.data.param2 && data.param3 == r.data.param3) return true;
			if (data.param1 == r.data.param2 && data.param2 == r.data.param1 && data.param3 == r.data.param3) return true;
			return false;

		case Expression::data::NodeType::COND:
			return data.param1 == r.data.param1 && data.param2 == r.data.param2 && data.param3 == r.data.param3;

		default:
			return data.param1 == r.data.param1;
		}
	}
};

//----------------------------------------------------------------------------------------------------------------------

template<> struct std::hash<Key>
{
	std::size_t operator()(Key const &r) const noexcept
	{
		size_t result = 0;
		result ^= std::hash<Expression::data::NodeType>{}(r.nodeType);
		result ^= std::hash<Expression::data const *>{}(r.data.param1);
		result ^= std::hash<Expression::data const *>{}(r.data.param2);
		result ^= std::hash<Expression::data const *>{}(r.data.param3);
		return result;
	}
};

//----------------------------------------------------------------------------------------------------------------------

inline std::unordered_map<Key, Expression::data const *> &nodeCache()
{
	static std::unordered_map<Key, Expression::data const *> result;
	return result;
}

/***********************************************************************************************************************
*** Nan
***********************************************************************************************************************/

struct Nan final : public Expression::data, private ObjectGuard<Nan>
{
	Nan() : Expression::data(NodeType::NOT_A_NUMBER) { }

	Expression abs_() const override final { return abs(Literal::nan); }
	Expression sgn_() const override final { return sgn(Literal::nan); }
	Expression sqrt_() const override final { return sqrt(Literal::nan); }
	Expression cbrt_() const override final { return cbrt(Literal::nan); }
	Expression exp_() const override final { return exp(Literal::nan); }
	Expression expm1_() const override final { return expm1(Literal::nan); }
	Expression log_() const override final { return log(Literal::nan); }
	Expression log1p_() const override final { return log1p(Literal::nan); }
	Expression sin_() const override final { return sin(Literal::nan); }
	Expression cos_() const override final { return cos(Literal::nan); }
	Expression tan_() const override final { return tan(Literal::nan); }
	Expression asin_() const override final { return asin(Literal::nan); }
	Expression acos_() const override final { return acos(Literal::nan); }
	Expression atan_() const override final { return atan(Literal::nan); }
	Expression sinh_() const override final { return sinh(Literal::nan); }
	Expression cosh_() const override final { return cosh(Literal::nan); }
	Expression tanh_() const override final { return tanh(Literal::nan); }
	Expression asinh_() const override final { return asinh(Literal::nan); }
	Expression acosh_() const override final { return acosh(Literal::nan); }
	Expression atanh_() const override final { return atanh(Literal::nan); }
	Expression erf_() const override final { return erf(Literal::nan); }
	Expression erfc_() const override final { return erfc(Literal::nan); }
	Expression ceil_() const override final { return ceil(Literal::nan); }
	Expression floor_() const override final { return floor(Literal::nan); }
	Expression invert_() const override final { return 1 / Literal::nan; }
	Expression negate_() const override final { return -Literal::nan; }
	Expression square_() const override final { return Literal::nan * Literal::nan; }

	Expression bind(Bindings const &) const override final { return this_(); }
	Expression derive(Variable const &) const override final { return this_(); }
	double compute() const noexcept override final { return Literal::nan; }

	void print(std::ostream &out, Context) const override final
	{
		out << "<nan>";
	}

	virtual ~Nan()
	{
	}
};

/***********************************************************************************************************************
*** ConstantNode
***********************************************************************************************************************/

struct ConstantNode final : public Expression::data, private ObjectGuard<ConstantNode>
{
	explicit ConstantNode(double d) : Expression::data(NodeType::CONSTANT), n(d)
	{
		auto &node = nodeCache();
		auto const key = Key(n);
		assert(node.find(key) == node.end());
		node[key] = this;
	}

	Expression abs_() const override final { return abs(n); }
	Expression sgn_() const override final { return sgn(n); }
	Expression sqrt_() const override final { return sqrt(n); }
	Expression cbrt_() const override final { return cbrt(n); }
	Expression exp_() const override final { return exp(n); }
	Expression expm1_() const override final { return expm1(n); }
	Expression log_() const override final { return log(n); }
	Expression log1p_() const override final { return log1p(n); }
	Expression sin_() const override final { return sin(n); }
	Expression cos_() const override final { return cos(n); }
	Expression tan_() const override final { return tan(n); }
	Expression asin_() const override final { return asin(n); }
	Expression acos_() const override final { return acos(n); }
	Expression atan_() const override final { return atan(n); }
	Expression sinh_() const override final { return sinh(n); }
	Expression cosh_() const override final { return cosh(n); }
	Expression tanh_() const override final { return tanh(n); }
	Expression asinh_() const override final { return asinh(n); }
	Expression acosh_() const override final { return acosh(n); }
	Expression atanh_() const override final { return atanh(n); }
	Expression erf_() const override final { return erf(n); }
	Expression erfc_() const override final { return erfc(n); }
	Expression ceil_() const override final { return ceil(n); }
	Expression floor_() const override final { return floor(n); }
	Expression invert_() const override final { return 1 / n; }
	Expression negate_() const override final { return -n; }
	Expression square_() const override final { return n * n; }

	Expression add_(Expression const &r) const override final { return r->is(NodeType::CONSTANT) ? n + r->evaluate() : Expression::data::add_(r); }
	Expression mul_(Expression const &r) const override final { return r->is(NodeType::CONSTANT) ? n * r->evaluate() : Expression::data::mul_(r); }
	Expression pow_(Expression const &r) const override final { return r->is(NodeType::CONSTANT) ? pow(n, r->evaluate()) : Expression::data::pow_(r); }
	Expression atan2_(Expression const &r) const override final { return r->is(NodeType::CONSTANT) ? atan2(n, r->evaluate()) : Expression::data::atan2_(r); }
	Expression equal_(Expression const &r) const override final { return r->is(NodeType::CONSTANT) ? n == r->evaluate() : Expression::data::equal_(r); }
	Expression fmod_(Expression const &r) const override final { return r->is(NodeType::CONSTANT) ? fmod(n, r->evaluate()) : Expression::data::fmod_(r); }
	Expression hypot_(Expression const &r) const override final { return r->is(NodeType::CONSTANT) ? hypot(n, r->evaluate()) : Expression::data::hypot_(r); }
	Expression less_(Expression const &r) const override final { return r->is(NodeType::CONSTANT) ? n < r->evaluate() : Expression::data::less_(r); }
	Expression nextafter_(Expression const &r) const override final { return r->is(NodeType::CONSTANT) ? nextafter(n, r->evaluate()) : Expression::data::nextafter_(r); }

	Expression cond_(Expression const &r, Expression const &s) const override final { return n ? r : s; }

	Expression bind(Bindings const &) const override final { return this_(); }
	Expression derive(Variable const &) const override final { return 0; }
	double compute() const noexcept override final { return n; }

	void print(std::ostream &out, Context) const override final
	{
		out << n;
	}

private:
	virtual ~ConstantNode()
	{
		auto &node = nodeCache();
		auto const key = Key(n);
		assert(node.find(key) != node.end() && node[key] == this);
		node.erase(key);
	}

	double const n;
};

/***********************************************************************************************************************
*** VariableNode
***********************************************************************************************************************/

struct VariableNode final : public Expression::data, private ObjectGuard<VariableNode>
{
	explicit VariableNode(Variable const &r) : Expression::data(NodeType::VARIABLE), x(r)
	{
		auto &node = nodeCache();
		auto const key = Key(x.id());
		assert(node.find(key) == node.end());
		node[key] = this;
	}

	Expression bind(Bindings const &r) const override final
	{
		if (!cache)
		{
			auto const node = r.bindings.find(x.id());
			Expression const result = node == r.bindings.end() ? this_() : node->second->function;
			cache = Clone(result.pData);
			return result;
		}
		return Clone(cache);
	}

	Expression derive(Variable const &r) const override final { return r.is(x); }
	double compute() const noexcept override final { return x->evaluate(); }

	void print(std::ostream &out, Context) const override final
	{
		out << x.Name();
	}

private:
	virtual ~VariableNode()
	{
		auto &node = nodeCache();
		auto const key = Key(x.id());
		assert(node.find(key) != node.end() && node[key] == this);
		node.erase(key);
	}

	Variable const x;
};

/***********************************************************************************************************************
*** FunctionNode
***********************************************************************************************************************/

struct FunctionNode : public Expression::data
{
	FunctionNode(NodeType n, Expression const &r) : Expression::data(n), x(r)
	{
		auto &node = nodeCache();
		auto const key = Key(nodeType, x.pData);
		assert(node.find(key) == node.end());
		node[key] = this;
	}

protected:
	Expression bind(Bindings const &r) const override final { if (!cache) cache = x->bind(r)->create(nodeType); return Clone(cache); }
	Expression derive(Variable const &r) const override final { if (!cache) cache = Clone((derive() * x->derive(r)).pData); return Clone(cache); }
	void flush() const noexcept override final { if (cache) { Expression::data::flush(); x->flush(); } }

	virtual Expression derive() const = 0;

	virtual ~FunctionNode()
	{
		auto &node = nodeCache();
		auto const key = Key(nodeType, x.pData);
		assert(node.find(key) != node.end() && node[key] == this);
		node.erase(key);
	}

	Expression const x;
};

/***********************************************************************************************************************
*** OperatorNode
***********************************************************************************************************************/

struct OperatorNode : public Expression::data
{
	OperatorNode(NodeType n, Expression const &r, Expression const &s) : Expression::data(n), f(r), g(s)
	{
		auto &node = nodeCache();
		auto const key = Key(nodeType, f.pData, g.pData);
		assert(node.find(key) == node.end());
		node[key] = this;
	}

protected:
	Expression bind(Bindings const &r) const override final { if (!cache) cache = f->bind(r)->create(nodeType, g->bind(r)); return Clone(cache); }
	void flush() const noexcept override final { if (cache) { Expression::data::flush(); f->flush(); g->flush(); } }

	virtual ~OperatorNode()
	{
		auto &node = nodeCache();
		auto const key = Key(nodeType, f.pData, g.pData);
		assert(node.find(key) != node.end() && node[key] == this);
		node.erase(key);
	}

	Expression const f;
	Expression const g;
};

/***********************************************************************************************************************
*** TernaryNode
***********************************************************************************************************************/

struct TernaryNode : public Expression::data
{
	TernaryNode(NodeType n, Expression const &r, Expression const &s, Expression const &t) : Expression::data(n), f(r), g(s), h(t)
	{
		auto &node = nodeCache();
		auto const key = Key(nodeType, f.pData, g.pData, h.pData);
		assert(node.find(key) == node.end());
		node[key] = this;
	}

protected:
	Expression bind(Bindings const &r) const override final { if (!cache) cache = f->bind(r)->create(nodeType, g->bind(r), h->bind(r)); return Clone(cache); }
	void flush() const noexcept override final { if (cache) { Expression::data::flush(); f->flush(); g->flush(); h->flush(); } }

	virtual ~TernaryNode()
	{
		auto &node = nodeCache();
		auto const key = Key(nodeType, f.pData, g.pData, h.pData);
		assert(node.find(key) != node.end() && node[key] == this);
		node.erase(key);
	}

	Expression const f;
	Expression const g;
	Expression const h;
};

/*
*	Template code for adding optimizations to various nodes:
* 
Expression abs_() const override final { return Expression::data::abs_(); }
Expression sgn_() const override final { return Expression::data::sgn_(); }
Expression sqrt_() const override final { return Expression::data::sqrt_(); }
Expression cbrt_() const override final { return Expression::data::cbrt_(); }
Expression exp_() const override final { return Expression::data::exp_(); }
Expression expm1_() const override final { return Expression::data::expm1_(); }
Expression log_() const override final { return Expression::data::log_(); }
Expression log1p_() const override final { return Expression::data::log1p_(); }
Expression sin_() const override final { return Expression::data::sin_(); }
Expression cos_() const override final { return Expression::data::cos_(); }
Expression tan_() const override final { return Expression::data::tan_(); }
Expression asin_() const override final { return Expression::data::asin_(); }
Expression acos_() const override final { return Expression::data::acos_(); }
Expression atan_() const override final { return Expression::data::atan_(); }
Expression sinh_() const override final { return Expression::data::sinh_(); }
Expression cosh_() const override final { return Expression::data::cosh_(); }
Expression tanh_() const override final { return Expression::data::tanh_(); }
Expression asinh_() const override final { return Expression::data::asinh_(); }
Expression acosh_() const override final { return Expression::data::acosh_(); }
Expression atanh_() const override final { return Expression::data::atanh_(); }
Expression erf_() const override final { return Expression::data::erf_(); }
Expression erfc_() const override final { return Expression::data::erfc_(); }
Expression ceil_() const override final { return Expression::data::ceil_(); }
Expression floor_() const override final { return Expression::data::floor_(); }
Expression invert_() const override final { return Expression::data::invert_(); }
Expression negate_() const override final { return Expression::data::negate_(); }
Expression square_() const override final { return Expression::data::square_(); }
*/

/***********************************************************************************************************************
*** Abs
***********************************************************************************************************************/

struct Abs final : public FunctionNode, private ObjectGuard<Abs>
{
	Abs(Expression const &r) : FunctionNode(NodeType::ABS, r) { }

	Expression abs_() const override final { return this_(); }

	Expression derive() const override final { return sgn(x); }
	double compute() const noexcept override final { return abs(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "abs(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** Sgn
***********************************************************************************************************************/

struct Sgn final : public FunctionNode, private ObjectGuard<Sgn>
{
	Sgn(Expression const &r) : FunctionNode(NodeType::SGN, r) { }

	Expression derive() const override final { return 0; }
	double compute() const noexcept override final { return sgn(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "sgn(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** Sqrt
***********************************************************************************************************************/

struct Sqrt final : public FunctionNode, private ObjectGuard<Sqrt>
{
	Sqrt(Expression const &r) : FunctionNode(NodeType::SQRT, r) { }

	Expression square_() const override final { return x; }

	Expression derive() const override final { return sqrt(x) / (x * 2); }
	double compute() const noexcept override final { return sqrt(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "sqrt(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** Cbrt
***********************************************************************************************************************/

struct Cbrt final : public FunctionNode, private ObjectGuard<Cbrt>
{
	Cbrt(Expression const &r) : FunctionNode(NodeType::CBRT, r) { }

	Expression derive() const override final { return cbrt(x) / (x * 3); }
	double compute() const noexcept override final { return cbrt(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "cbrt(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** Exp
***********************************************************************************************************************/

struct Exp final : public FunctionNode, private ObjectGuard<Exp>
{
	Exp(Expression const &r) : FunctionNode(NodeType::EXP, r) { }

	Expression abs_() const override final { return this_(); }
	Expression log_() const override final { return x; }

	Expression derive() const override final { return this_(); }
	double compute() const noexcept override final { return exp(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "exp(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** ExpM1
***********************************************************************************************************************/

struct ExpM1 final : public FunctionNode, private ObjectGuard<ExpM1>
{
	ExpM1(Expression const &r) : FunctionNode(NodeType::EXPM1, r) { }

	Expression log1p_() const override final { return x; }

	Expression derive() const override final { return exp(x); }
	double compute() const noexcept override final { return expm1(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "expm1(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** Log
***********************************************************************************************************************/

struct Log final : public FunctionNode, private ObjectGuard<Log>
{
	Log(Expression const &r) : FunctionNode(NodeType::LOG, r) { }

	Expression exp_() const override final { return x; }

	Expression derive() const override final { return 1 / x; }
	double compute() const noexcept override final { return log(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "log(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** Log1P
***********************************************************************************************************************/

struct Log1P final : public FunctionNode, private ObjectGuard<Log1P>
{
	Log1P(Expression const &r) : FunctionNode(NodeType::LOG1P, r) { }

	Expression expm1_() const override final { return x; }

	Expression derive() const override final { return 1 / (x + 1); }
	double compute() const noexcept override final { return log1p(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "log1p(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** Sin
***********************************************************************************************************************/

struct Sin final : public FunctionNode, private ObjectGuard<Sin>
{
	Sin(Expression const &r) : FunctionNode(NodeType::SIN, r) { }

	Expression derive() const override final { return cos(x); }
	double compute() const noexcept override final { return sin(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "sin(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** Cos
***********************************************************************************************************************/

struct Cos final : public FunctionNode, private ObjectGuard<Cos>
{
	Cos(Expression const &r) : FunctionNode(NodeType::COS, r) { }

	Expression derive() const override final { return -sin(x); }
	double compute() const noexcept override final { return cos(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "cos(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** Tan
***********************************************************************************************************************/

struct Tan final : public FunctionNode, private ObjectGuard<Tan>
{
	Tan(Expression const &r) : FunctionNode(NodeType::TAN, r) { }

	Expression derive() const override final { return 1 / (cos(x) * cos(x)); }
	double compute() const noexcept override final { return tan(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "tan(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** ASin
***********************************************************************************************************************/

struct ASin final : public FunctionNode, private ObjectGuard<ASin>
{
	ASin(Expression const &r) : FunctionNode(NodeType::ASIN, r) { }

	Expression sin_() const override final { return x; }

	Expression derive() const override final { return 1 / sqrt(1 - x * x); }
	double compute() const noexcept override final { return asin(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "asin(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** ACos
***********************************************************************************************************************/

struct ACos final : public FunctionNode, private ObjectGuard<ACos>
{
	ACos(Expression const &r) : FunctionNode(NodeType::ACOS, r) { }

	Expression cos_() const override final { return x; }

	Expression abs_() const override final { return this_(); }

	Expression derive() const override final { return -1 / sqrt(1 - x * x); }
	double compute() const noexcept override final { return acos(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "acos(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** ATan
***********************************************************************************************************************/

struct ATan final : public FunctionNode, private ObjectGuard<ATan>
{
	ATan(Expression const &r) : FunctionNode(NodeType::ATAN, r) { }

	Expression tan_() const override final { return x; }

	Expression derive() const override final { return 1 / (1 + x * x); }
	double compute() const noexcept override final { return atan(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "atan(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** SinH
***********************************************************************************************************************/

struct SinH final : public FunctionNode, private ObjectGuard<SinH>
{
	SinH(Expression const &r) : FunctionNode(NodeType::SINH, r) { }

	Expression asinh_() const override final { return x; }

	Expression derive() const override final { return cosh(x); }
	double compute() const noexcept override final { return sinh(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "sinh(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** CosH
***********************************************************************************************************************/

struct CosH final : public FunctionNode, private ObjectGuard<CosH>
{
	CosH(Expression const &r) : FunctionNode(NodeType::COSH, r) { }

	Expression abs_() const override final { return this_(); }
	Expression acosh_() const override final { return abs(x); }

	Expression derive() const override final { return sinh(x); }
	double compute() const noexcept override final { return cosh(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "cosh(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** TanH
***********************************************************************************************************************/

struct TanH final : public FunctionNode, private ObjectGuard<TanH>
{
	TanH(Expression const &r) : FunctionNode(NodeType::TANH, r) { }

	Expression atanh_() const override final { return x; }

	Expression derive() const override final { return 1 / (cosh(x) * cosh(x)); }
	double compute() const noexcept override final { return tanh(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "tanh(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** ASinH
***********************************************************************************************************************/

struct ASinH final : public FunctionNode, private ObjectGuard<ASinH>
{
	ASinH(Expression const &r) : FunctionNode(NodeType::ASINH, r) { }

	Expression sinh_() const override final { return x; }

	Expression derive() const override final { return 1 / sqrt(x * x + 1); }
	double compute() const noexcept override final { return asinh(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "asinh(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** ACosH
***********************************************************************************************************************/

struct ACosH final : public FunctionNode, private ObjectGuard<ACosH>
{
	ACosH(Expression const &r) : FunctionNode(NodeType::ACOSH, r) { }

	Expression cosh_() const override final { return x; }

	Expression derive() const override final { return 1 / sqrt(x - 1) / sqrt(x + 1); }
	double compute() const noexcept override final { return acosh(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "acosh(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** ATanH
***********************************************************************************************************************/

struct ATanH final : public FunctionNode, private ObjectGuard<ATanH>
{
	ATanH(Expression const &r) : FunctionNode(NodeType::ATANH, r) { }

	Expression tanh_() const override final { return x; }

	Expression derive() const override final { return 1 / (1 - x) / (1 + x); }
	double compute() const noexcept override final { return atanh(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "atanh(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** Erf
***********************************************************************************************************************/

struct Erf final : public FunctionNode, private ObjectGuard<Erf>
{
	Erf(Expression const &r) : FunctionNode(NodeType::ERF, r) { }

	Expression derive() const override final { return 2 / sqrt(acos(-1)) * exp(-x * x); }
	double compute() const noexcept override final { return erf(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "erf(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** ErfC
***********************************************************************************************************************/

struct ErfC final : public FunctionNode, private ObjectGuard<ErfC>
{
	ErfC(Expression const &r) : FunctionNode(NodeType::ERFC, r) { }

	Expression abs_() const override final { return this_(); }

	Expression derive() const override final { return -2 / sqrt(acos(-1)) * exp(-x * x); }
	double compute() const noexcept override final { return erfc(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "erfc(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** Ceil
***********************************************************************************************************************/

struct Ceil final : public FunctionNode, private ObjectGuard<Ceil>
{
	Ceil(Expression const &r) : FunctionNode(NodeType::CEIL, r) { }

	Expression derive() const override final { return 0; }
	double compute() const noexcept override final { return ceil(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "ceil(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** Floor
***********************************************************************************************************************/

struct Floor final : public FunctionNode, private ObjectGuard<Floor>
{
	Floor(Expression const &r) : FunctionNode(NodeType::FLOOR, r) { }

	Expression derive() const override final { return 0; }
	double compute() const noexcept override final { return floor(x->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "floor(";
		x->print(out, Context::DEFAULT);
		out << ")";
	}
};

/***********************************************************************************************************************
*** Invert
***********************************************************************************************************************/

struct Invert final : public FunctionNode, private ObjectGuard<Invert>
{
	Invert(Expression const &r) : FunctionNode(NodeType::INVERT, r) { }

	Expression abs_() const override final { return 1 / abs(x); }
	Expression invert_() const override final { return x; }

	Expression derive() const override final { return -1 / (x * x); }
	double compute() const noexcept override final { return 1 / x->evaluate(); }

	void print(std::ostream &out, Context) const override final
	{
		out << "1 / ";
		if (x->is(NodeType::ADD) || x->is(NodeType::MUL) || x->is(NodeType::NEGATE)) out << "(";
		x->print(out);
		if (x->is(NodeType::ADD) || x->is(NodeType::MUL) || x->is(NodeType::NEGATE)) out << ")";
	}
};

/***********************************************************************************************************************
*** Negate
***********************************************************************************************************************/

struct Negate final : public FunctionNode, private ObjectGuard<Negate>
{
	Negate(Expression const &r) : FunctionNode(NodeType::NEGATE, r) { }

	Expression cbrt_() const override final { return -cbrt(x); }
	Expression negate_() const override final { return x; }

	Expression mul_(Expression const &r) const override final { return (r->is(NodeType::CONSTANT) || r->is(NodeType::NEGATE)) ? x * (-r) : -(x * r); }

	Expression derive() const override final { return -1; }
	double compute() const noexcept override final { return -x->evaluate(); }

	void print(std::ostream &out, Context) const override final
	{
		out << "-";
		x->print(out, Context::NEGATE);
	}
};

/***********************************************************************************************************************
*** Square
***********************************************************************************************************************/

struct Square final : public FunctionNode, private ObjectGuard<Square>
{
	Square(Expression const &r) : FunctionNode(NodeType::SQUARE, r) { }

	Expression abs_() const override final { return this_(); }
	Expression sqrt_() const override final { return abs(x); }

	Expression derive() const override final { return x * 2; }
	double compute() const noexcept override final { auto const t = x->evaluate(); return t * t; }

	void print(std::ostream &out, Context) const override final
	{
		if (x->is(NodeType::ADD) || x->is(NodeType::MUL) || x->is(NodeType::NEGATE)) out << "(";
		x->print(out);
		if (x->is(NodeType::ADD) || x->is(NodeType::MUL) || x->is(NodeType::NEGATE)) out << ")";
		out << "^2";
	}
};

/***********************************************************************************************************************
*** Add
***********************************************************************************************************************/

struct Add final : public OperatorNode, private ObjectGuard<Add>
{
	Add(Expression const &r, Expression const &s) : OperatorNode(NodeType::ADD, r, s) { }

	Expression derive(Variable const &r) const override final { if (!cache) cache = Clone((f->derive(r) + g->derive(r)).pData); return Clone(cache); }
	double compute() const noexcept override final { return f->evaluate() + g->evaluate(); }

	void print(std::ostream &out, Context c) const override final
	{
		if (c == Context::INVERT || c == Context::NEGATE || c == Context::MUL) out << "(";
		f->print(out, Context::ADD);
		out << " + ";
		g->print(out, Context::ADD);
		if (c == Context::INVERT || c == Context::NEGATE || c == Context::MUL) out << ")";
	}

private:
	virtual ~Add()
	{
	}
};

/***********************************************************************************************************************
*** Mul
***********************************************************************************************************************/

struct Mul final : public OperatorNode, private ObjectGuard<Mul>
{
	Mul(Expression const &r, Expression const &s) : OperatorNode(NodeType::MUL, r, s) { }

	Expression derive(Variable const &r) const override final { if (!cache) cache = Clone((f->derive(r) * g + g->derive(r) * f).pData); return Clone(cache); }
	double compute() const noexcept override final { return f->evaluate() * g->evaluate(); }

	void print(std::ostream &out, Context c) const override final
	{
		if (c == Context::INVERT) out << "(";
		f->print(out, Context::MUL);
		out << " * ";
		g->print(out, Context::MUL);
		if (c == Context::INVERT) out << ")";
	}

private:
	virtual ~Mul()
	{
	}
};

/***********************************************************************************************************************
*** Pow
***********************************************************************************************************************/

struct Pow final : public OperatorNode, private ObjectGuard<Pow>
{
	Pow(Expression const &r, Expression const &s) : OperatorNode(NodeType::POW, r, s) { }

	Expression derive(Variable const &r) const override final { if (!cache) cache = Clone((pow(f, g) * (f->derive(r) * g / f + g->derive(r) * log(f))).pData); return Clone(cache); }
	double compute() const noexcept override final { return pow(f->evaluate(), g->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "pow(";
		f->print(out, Context::DEFAULT);
		out << ", ";
		g->print(out, Context::DEFAULT);
		out << ")";
	}

private:
	virtual ~Pow()
	{
	}
};

/***********************************************************************************************************************
*** ATan2
***********************************************************************************************************************/

struct ATan2 final : public OperatorNode, private ObjectGuard<ATan2>
{
	ATan2(Expression const &r, Expression const &s) : OperatorNode(NodeType::ATAN2, r, s) { }

	Expression derive(Variable const &r) const override final { if (!cache) cache = Clone(((f->derive(r) * g - g->derive(r) * f) / (f * f + g * g)).pData); return Clone(cache); }
	double compute() const noexcept override final { return atan2(f->evaluate(), g->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "atan2(";
		f->print(out, Context::DEFAULT);
		out << ", ";
		g->print(out, Context::DEFAULT);
		out << ")";
	}

private:
	virtual ~ATan2()
	{
	}
};

/***********************************************************************************************************************
*** Equal
***********************************************************************************************************************/

struct Equal final : public OperatorNode, private ObjectGuard<Equal>
{
	Equal(Expression const &r, Expression const &s) : OperatorNode(NodeType::EQUAL, r, s) { }

	Expression derive(Variable const &) const override final { return 0; }
	double compute() const noexcept override final { return f->evaluate() == g->evaluate(); }

	void print(std::ostream &out, Context) const override final
	{
		out << "Equal(";
		f->print(out, Context::DEFAULT);
		out << ", ";
		g->print(out, Context::DEFAULT);
		out << ")";
	}

private:
	virtual ~Equal()
	{
	}
};

/***********************************************************************************************************************
*** FMod
***********************************************************************************************************************/

struct FMod final : public OperatorNode, private ObjectGuard<FMod>
{
	FMod(Expression const &r, Expression const &s) : OperatorNode(NodeType::FMOD, r, s) { }

	Expression derive(Variable const &r) const override final { if (!cache) cache = Clone((f->derive(r) - g->derive(r) * floor(f / g)).pData); return Clone(cache); }
	double compute() const noexcept override final { return fmod(f->evaluate(), g->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "fmod(";
		f->print(out, Context::DEFAULT);
		out << ", ";
		g->print(out, Context::DEFAULT);
		out << ")";
	}

private:
	virtual ~FMod()
	{
	}
};

/***********************************************************************************************************************
*** Hypot
***********************************************************************************************************************/

struct Hypot final : public OperatorNode, private ObjectGuard<Hypot>
{
	Hypot(Expression const &r, Expression const &s) : OperatorNode(NodeType::HYPOT, r, s) { }

	Expression abs_() const override final { return this_(); }

	Expression derive(Variable const &r) const override final { if (!cache) cache = Clone(((f->derive(r) * f + g->derive(r) * g) / hypot(f, g)).pData); return Clone(cache); }
	double compute() const noexcept override final { return hypot(f->evaluate(), g->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "hypot(";
		f->print(out, Context::DEFAULT);
		out << ", ";
		g->print(out, Context::DEFAULT);
		out << ")";
	}

private:
	virtual ~Hypot()
	{
	}
};

/***********************************************************************************************************************
*** Less
***********************************************************************************************************************/

struct Less final : public OperatorNode, private ObjectGuard<Less>
{
	Less(Expression const &r, Expression const &s) : OperatorNode(NodeType::LESS, r, s) { }

	Expression derive(Variable const &) const override final { return 0; }
	double compute() const noexcept override final { return f->evaluate() < g->evaluate(); }

	void print(std::ostream &out, Context) const override final
	{
		out << "(";
		f->print(out, Context::DEFAULT);
		out << "<";
		g->print(out, Context::DEFAULT);
		out << ")";
	}

private:
	virtual ~Less()
	{
	}
};

/***********************************************************************************************************************
*** NextAfter
***********************************************************************************************************************/

struct NextAfter final : public OperatorNode, private ObjectGuard<NextAfter>
{
	NextAfter(Expression const &r, Expression const &s) : OperatorNode(NodeType::NEXTAFTER, r, s) { }

	Expression derive(Variable const &r) const override final { if (!cache) cache = Clone((f->derive(r)).pData); return Clone(cache); }
	double compute() const noexcept override final { return nextafter(f->evaluate(), g->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "nextafter(";
		f->print(out, Context::DEFAULT);
		out << ",";
		g->print(out, Context::DEFAULT);
		out << ")";
	}

private:
	virtual ~NextAfter()
	{
	}
};

/***********************************************************************************************************************
*** Cond
***********************************************************************************************************************/

struct Cond : public TernaryNode, private ObjectGuard<Cond>
{
	Cond(Expression const &r, Expression const &s, Expression const &t) : TernaryNode(NodeType::COND, r, s, t) { }

	Expression derive(Variable const &r) const override final { if (!cache) cache = Clone((cond(f, g->derive(r), h->derive(r))).pData); return Clone(cache); }
	double compute() const noexcept override final { return f->evaluate() == 0 ? h->evaluate() : g->evaluate(); }

	void print(std::ostream &out, Context) const override final
	{
		out << "cond(";
		f->print(out, Context::DEFAULT);
		out << ", ";
		g->print(out, Context::DEFAULT);
		out << ", ";
		h->print(out, Context::DEFAULT);
		out << ")";
	}

private:
	virtual ~Cond()
	{
	}
};

/***********************************************************************************************************************
*** Fma
***********************************************************************************************************************/

struct Fma : public TernaryNode, private ObjectGuard<Fma>
{
	Fma(Expression const &r, Expression const &s, Expression const &t) : TernaryNode(NodeType::FMA, r, s, t) { }

	Expression derive(Variable const &r) const override final { if (!cache) cache = Clone(fma(f->derive(r), g, fma(g->derive(r), f, h->derive(r))).pData); return Clone(cache); }
	double compute() const noexcept override final { return fma(f->evaluate(), g->evaluate(), h->evaluate()); }

	void print(std::ostream &out, Context) const override final
	{
		out << "fma(";
		f->print(out, Context::DEFAULT);
		out << ", ";
		g->print(out, Context::DEFAULT);
		out << ", ";
		h->print(out, Context::DEFAULT);
		out << ")";
	}

private:
	virtual ~Fma()
	{
	}
};

/***********************************************************************************************************************
*** create(double)
***********************************************************************************************************************/

inline Expression::data const *Expression::data::create(double d)
{
	if (isnan(d))
	{
		static Nan const result;
		return Clone(result);
	}

	auto const node = nodeCache().find(Key(d));
	return node != nodeCache().end() ? Clone(node->second) : new ConstantNode(d);
}

/***********************************************************************************************************************
*** create(Variable const &)
***********************************************************************************************************************/

inline Expression::data const *Expression::data::create(Variable const &r)
{
	auto const node = nodeCache().find(Key(r.id()));
	return node != nodeCache().end() ? Clone(node->second) : new VariableNode(r);
}

/***********************************************************************************************************************
*** create(NodeType)
***********************************************************************************************************************/

inline Expression::data const *Expression::data::create(NodeType n) const
{
	auto const node = nodeCache().find(Key(n, this));
	if (node != nodeCache().end()) return Clone(node->second);

	switch (n)
	{
	case NodeType::ABS:
		return new Abs(this_());

	case NodeType::SGN:
		return new Sgn(this_());

	case NodeType::SQRT:
		return new Sqrt(this_());

	case NodeType::CBRT:
		return new Cbrt(this_());

	case NodeType::EXP:
		return new Exp(this_());

	case NodeType::EXPM1:
		return new ExpM1(this_());

	case NodeType::LOG:
		return new Log(this_());

	case NodeType::LOG1P:
		return new Log1P(this_());

	case NodeType::SIN:
		return new Sin(this_());

	case NodeType::COS:
		return new Cos(this_());

	case NodeType::TAN:
		return new Tan(this_());

	case NodeType::ASIN:
		return new ASin(this_());

	case NodeType::ACOS:
		return new ACos(this_());

	case NodeType::ATAN:
		return new ATan(this_());

	case NodeType::SINH:
		return new SinH(this_());

	case NodeType::COSH:
		return new CosH(this_());

	case NodeType::TANH:
		return new TanH(this_());

	case NodeType::ASINH:
		return new ASinH(this_());

	case NodeType::ACOSH:
		return new ACosH(this_());

	case NodeType::ATANH:
		return new ATanH(this_());

	case NodeType::ERF:
		return new Erf(this_());

	case NodeType::ERFC:
		return new ErfC(this_());

	case NodeType::CEIL:
		return new Ceil(this_());

	case NodeType::FLOOR:
		return new Floor(this_());

	case NodeType::INVERT:
		return new Invert(this_());

	case NodeType::NEGATE:
		return new Negate(this_());

	case NodeType::SQUARE:
		return new Square(this_());
	}

	UNREACHABLE;
}

/***********************************************************************************************************************
*** create(NodeType, Expression const &)
***********************************************************************************************************************/

inline Expression::data const *Expression::data::create(NodeType n, Expression const &r) const
{
	auto const node = nodeCache().find(Key(n, this, r.pData));
	if (node != nodeCache().end()) return Clone(node->second);

	switch (n)
	{
	case NodeType::ADD:
		return new Add(this_(), r);

	case NodeType::MUL:
		return new Mul(this_(), r);

	case NodeType::POW:
		return new Pow(this_(), r);

	case NodeType::ATAN2:
		return new ATan2(this_(), r);

	case NodeType::FMOD:
		return new FMod(this_(), r);

	case NodeType::EQUAL:
		return new Equal(this_(), r);

	case NodeType::HYPOT:
		return new Hypot(this_(), r);

	case NodeType::LESS:
		return new Less(this_(), r);

	case NodeType::NEXTAFTER:
		return new NextAfter(this_(), r);
	}

	UNREACHABLE;
}

/***********************************************************************************************************************
*** create(NodeType, Expression const &, Expression const &)
***********************************************************************************************************************/

inline Expression::data const *Expression::data::create(NodeType n, Expression const &r, Expression const &s) const
{
	auto const node = nodeCache().find(Key(n, this, r.pData, s.pData));
	if (node != nodeCache().end()) return Clone(node->second);

	switch (n)
	{
	case NodeType::COND:
		return new Cond(this_(), r, s);

	case NodeType::FMA:
		return new Fma(this_(), r, s);
	}

	UNREACHABLE;
}

/***********************************************************************************************************************
*** Expression::data
***********************************************************************************************************************/

inline Expression Expression::data::add_(Expression const &r) const
{
	if (r->is(NodeType::CONSTANT) && r->evaluate() == 0) return this_();
	if (r.pData == this) return mul_(2);
	return create(NodeType::ADD, r);
}

inline Expression Expression::data::mul_(Expression const &r) const
{
	if (r->is(NodeType::CONSTANT))
	{
		auto const d = r->evaluate();
		if (d == 1) return this_();
		if (d == 0) return 0;
		if (d == -1) return negate_();
	}
	if (r.pData == this) return square_();
	return create(NodeType::MUL, r);
}

inline Expression Expression::data::pow_(Expression const &r) const
{
	if (r->is(NodeType::CONSTANT))
	{
		auto const d = r->evaluate();
		if (d == 2) return square_();
		if (d == 1) return this_();
		if (d == 0) return 1;
		if (d == -1) return invert_();
		if (d == 0.5) return sqrt_();
	}
	return create(NodeType::POW, r);
}

inline Expression Expression::data::hypot_(Expression const &r) const
{
	if (r->is(NodeType::CONSTANT) && r->evaluate() == 0) return this_();
	return create(NodeType::HYPOT, r);
}

/***********************************************************************************************************************
*** Expression
***********************************************************************************************************************/

inline Expression::Expression() : pData(data::create(0))
{
}

inline Expression::Expression(Expression const &r) noexcept : pData(Shared::Clone(r.pData))
{
}

inline Expression::Expression(double d) : pData(data::create(d))
{
}

inline Expression::Expression(Variable const &r) : pData(data::create(r))
{
}

inline Expression::Expression(data const *p) noexcept : pData(p)
{
}

inline Expression::~Expression()
{
	Shared::Erase(pData);
}

inline Expression &Expression::operator=(Expression const &r) noexcept
{
	if (pData != r.pData)
	{
		Shared::Erase(pData);
		pData = Shared::Clone(r.pData);
	}
	return *this;
}

inline Expression Expression::Bind(Bindings const &r) const
{
	Expression const result = pData->bind(r);
	pData->flush();
	return result;
}

inline Expression Expression::Derive(Variable const &r) const
{
	Expression const result = pData->derive(r);
	pData->flush();
	return result;
}

inline std::vector<Expression> Expression::Derive(Variable const &r, int n) const
{
	assert(n >= 0);
	std::vector<Expression> result(n + 1ULL);
	result[0] = *this;
	for (int i = 1; i <= n; ++i) result[i] = result[i - 1ULL].pData->derive(r);
	for (int i = 0; i <= n; ++i) result[i].pData->flush();
	return result;
}
