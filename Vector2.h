#pragma once

template<typename T>
class Vector2
{
private:
	T _x, _y;
public:
	Vector2(T x = 0, T y = 0)
		:_x(x), _y(y)
	{}

	float getLength() const 
	{
		return sqrt(_x*_x*1.0 + _y*_y*1.0);
	}

	void x(T x)
	{
		_x = x;
	}

	T x() const
	{
		return _x;
	}

	void y(T y)
	{
		_y = y;
	}

	T y() const
	{
		return _y;
	}

	void operator /=(float m)
	{
		if (abs(m)<=0.001f)
			return;
		this->_x /= m;
		this->_y /= m;
	}

	Vector2<T> operator -(const Vector2<T> right)
	{
		return Vector2<T>(this->_x - right._x, this->_y - right._y);
	}

	template<typename E>
	float dot(const Vector2<E>& right){
		return this->_x*right.x() + this->_y*right.y();
	}
};