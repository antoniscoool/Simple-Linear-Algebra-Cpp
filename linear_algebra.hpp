#include <type_traits>
#include <array>
#include <functional>
#include <iostream>
#include <algorithm>


using mat_dimension_type = std::uint32_t;




// Equal operation used in the requires statements
template<size_t A, size_t B>
inline constexpr bool _equal = A == B;

// Greater or equal than operation used in the requires statements
template<size_t A, size_t B>
inline constexpr bool _greater_or_equal = A >= B;

// General macro for the operator overloads.
// op: Operator to use
// f: Neutral value of the operator (a op n = a), in case loop(s) go out of bounds of the other matrix
#define MAKE_OPERATION(op,f)	template<Numeric _T, mat_dimension_type _W, mat_dimension_type _H> \
								matrix<T, W, H> operator op (const matrix<_T, _W, _H>& other) requires _greater_or_equal<W,_W> and _greater_or_equal<H,_H> \
								{ \
									matrix<T, W, H> res = *this; \
									for (mat_dimension_type i = 0; i < H; i++) \
										for (mat_dimension_type j = 0; j < W; j++) \
											res.data[j + i * W] op##= (j >= _W || i >= _H) ? (f) : other.data[j + i * W]; \
									return res; \
								} \
								matrix<T, W, H> operator##op##(const T val) \
								{ \
									matrix<T, W, H> res = *this; \
									for (mat_dimension_type i = 0; i < H; i++) \
										for (mat_dimension_type j = 0; j < W; j++) \
											res.data[j + i * W] op##= val; \
									return res; \
								} 


// Defines the numeric type concept, a generic type which only accepts numeric types
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

/*
	Class for the average matrix.
	
	Generic parameters:
		- T: any numeric type (int, float etc.)
		- W: column count (width)
		- H: row count (height)
*/
template<Numeric T, mat_dimension_type W=1, mat_dimension_type H=1>
class matrix
{
public:

	std::array<T, W * H> data;

	// Fills the matrix' data array with either zeros or a single number
	// (if provided)
	matrix() { data.fill((T)0); }
	matrix(const T val) { data.fill( (T)val); }
	
	// Fills the matrix' data array with the values inside parameter pack
	template<Numeric... _T>
	matrix(const _T... vals) : data{ T(vals)...} {}

	// Destructor, not used
	~matrix() {}
	
	// Operator overloadings for componentwise operations (provided as macros)
	// NOTE: Only works when the other matrix has smaller or equal dimensions than this one
	MAKE_OPERATION(+, 0);
	MAKE_OPERATION(-, 0);
	MAKE_OPERATION(*, 1);
	MAKE_OPERATION(/, 1);

	
	template<Numeric _T>
	matrix<_T, W, H> CastToNum()
	{
		matrix<_T, W, H> out;
		for (mat_dimension_type i = 0; i < data.size(); i++) out.data[i] = static_cast<_T>(data[i]);
		return out;
	}

	// Applies lambda function to each matrix cell
	// NOTE: Should be in the form T(T) where T is a numeric type
	matrix<T, W, H> ApplyFunction(const auto func) requires std::is_invocable_v<decltype(func), T>
	{
		for (auto& element : data) {
			element = func(element);
		}
		return *this;
	}

	// Performs matrix multiplication
	// NOTE: the row count of matrix A should be equal to the column count of matrix B
	// Use of auto keyword when storing recommended (auto C = A.mat_mul(B))
	template<Numeric _T, mat_dimension_type W2, mat_dimension_type H2>
	matrix<T, W2, H> Mat_mul(const matrix<_T, W2, H2>& o) requires _equal<W, H2>
	{
		matrix<T, W2, H> res;
		for (mat_dimension_type i = 0; i < H; ++i)
		{
			for (mat_dimension_type j = 0; j < W2; ++j)
			{
				T sum = 0;
				for (unsigned k = 0; k < W; ++k) sum += data[i * W + k] * o.data[k * W2 + j];
				res.data[i * W2 + j] = sum;
			}
		}
		return res;
	}

	// Calculates the dot product
	// NOTE: A and B should be either column or row vectors of equal length
	float Dot(const matrix<T, W, H>& o) requires _equal<W, 1> or _equal<H, 1>
	{
		return [&]<std::size_t... p>(std::index_sequence<p...>)
		{
			return (float)((data[p] * o.data[p]) + ...);
		}(std::make_index_sequence<W* H>{});
	}

	// Calculates the cross product
	// NOTE:	1) A and B should be either column or row vectors of equal length
	//	2) A and B should have at least 3 elements
	matrix<T,W,H> Cross(const matrix<T,W,H>& o) requires (_greater_or_equal<W,3> and _equal<H,1>) 
													or (_greater_or_equal<H, 3> and _equal<W, 1>)
	{
		T	x = data[0], ox = o.data[0], 
			y = data[1], oy = o.data[1], 
			z = data[2], oz = o.data[2];

		matrix<T, W, H> result = *this;

		result.data[0] = y * oz - z * oy;
		result.data[1] = z * ox - x * oz;
		result.data[2] = x * oy - y * ox;

		return result;
	}

	// Converts a square matrix into a identity matrix
	matrix<T,W,H> Identity()
	{
		for (mat_dimension_type i = 0; i < H; ++i)
			for (mat_dimension_type j = 0; j < W; ++j)
				data[i * W + j] = i == j ? (T)1 : (T)0;
		return *this;
	}

	// Calculates the matrix that was linearly interpolated.
	// NOTE: see operator overloads
	template<Numeric _T, mat_dimension_type _W, mat_dimension_type _H>
	matrix<T, W, H> Lerp(matrix<_T, _W, _H>& o, float t, bool clamp_to_values = false) requires _greater_or_equal<W, _W> and _greater_or_equal<H, _H>
	{
		if(clamp_to_values) t = std::clamp(t, 0.0f, 1.0f);
		return (*this) * (1.0f - t) + o * t;
	}

	// Calculates the length/magnitude of the matrix
	// NOTE: 1) A and B should be either column or row vectors of equal length
	float Length() requires _equal<W, 1> or _equal<H, 1>
	{
		return std::sqrt(this->Dot(*this));
	}

	// Returns the normalized vector
	// NOTE: 1) A and B should be either column or row vectors of equal length
	matrix<T, W, H> Normalize() requires _equal<W, 1> or _equal<H, 1>
	{
		return *this / this->Length();
	}

	//Prints out the matrix into stdout
	void Print()
	{
		std::apply([&](const auto... args) {
			int i = 0;
			((std::cout << args << (++i % W == 0 ? "\n" : " ")), ...);
		}, data);
	}
};


// Below are type definitions

using vec2f = matrix<float, 1, 2>;

using vec2i = matrix<int, 1, 2>;


using vec3f = matrix<float, 1, 3>;

using vec3i = matrix<int, 1, 3>;


using vec4f = matrix<float, 1, 4>;

using vec4i = matrix<int, 1, 4>;


using mat3f = matrix<float, 3, 3>;

using mat3i = matrix<int, 3, 3>;


using mat4f = matrix<float, 4, 4>;

using mat4i = matrix<int, 4, 4>;


// TODO
class quart : vec4f {};
