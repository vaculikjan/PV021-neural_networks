/**
 * @brief A wrapper performing efficient matrix transposition.
 *
 * Wrapper converts rows to columns and vice versa. It avoids copying data and therefore provides a speedup.
 *
 * @param data Input matrix in the form of std::vector<std::vector<double>>.
 */

template<typename T>
class TwoDPivotWrapper
{
    public:
        // These two typedef's were done with std::vector
        // in mind. But with a small amount of effort I am
        // sure they can be generalized. They are solely to define
        // value_type (the data stored in the 2-D array).
        typedef typename T::value_type          OneDType;
        typedef typename OneDType::value_type   value_type;

        // A constructor that wraps a 2-D structure.
        TwoDPivotWrapper(T& o)
            : object(o)
        {}

        // A helper class used to store the row after the first array accesses.
        class Row
        {
            friend class TwoDPivotWrapper;
            Row(TwoDPivotWrapper& w, size_t r)
                : wrapper(w)
                , row(r)
            {}

            TwoDPivotWrapper&    wrapper;  
            size_t               row;

            public:
                value_type operator[](size_t col)
                {                    
                    return wrapper.get(row,col);
                }
        };

        // The operator [] returns a Row object that overloads
        // the operator[] for the next dimension.
        Row operator[](size_t row)              {return Row(*this, row);}

        // Generic get function used to access elements.
        // Notice we swap the row/col around when accessing
        // the underlying object.
        value_type get(size_t row, size_t col)  {return object[col][row];}

    public:
        T&  object;
};