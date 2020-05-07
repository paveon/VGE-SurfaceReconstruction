/***
 * millipede: DisjointSetForest.h
 * Copyright Stuart Golodetz, 2009. All rights reserved.
 ***/

// Slightly modified for C++11

#ifndef H_MILLIPEDE_DISJOINTSETFOREST
#define H_MILLIPEDE_DISJOINTSETFOREST

#include <unordered_map>


/**
@brief  A disjoint set forest is a fairly standard data structure used to represent the partition of
        a set of elements into disjoint sets in such a way that common operations such as merging two
        sets together are computationally efficient.

This implementation uses the well-known union-by-rank and path compression optimizations, which together
yield an amortised complexity for key operations of O(a(n)), where a is the (extremely slow-growing)
inverse of the Ackermann function.

The implementation also allows clients to attach arbitrary data to each element, which can be useful for
some algorithms.

@tparam T   The type of data to attach to each element (arbitrary)
*/
template<typename T = nullptr_t>
class DisjointSetForest {
    //#################### NESTED CLASSES ####################
private:
    struct Element {
        T m_value;
        size_t m_parent = 0;
        size_t m_rank = 0;

        Element() = default;

        Element(const T &value, size_t parent) : m_value(value), m_parent(parent), m_rank(0) {}
    };

    //#################### PRIVATE VARIABLES ####################
private:
    std::unordered_map<size_t, Element> m_elements;
    size_t m_setCount;

    //#################### CONSTRUCTORS ####################
public:
    /**
    @brief  Constructs an empty disjoint set forest.
    */
    DisjointSetForest() : m_setCount(0) {}

    /**
    @brief  Constructs a disjoint set forest from an initial set of elements and their associated values.

    @param[in]  initialElements     A map from the initial elements to their associated values
    */
    explicit DisjointSetForest(const std::unordered_map<size_t, T> &initialElements) : m_setCount(0) {
        add_elements(initialElements);
    }

    explicit DisjointSetForest(const std::vector<T>& initialElements) {
        for (m_setCount = 0; m_setCount < initialElements.size(); m_setCount++) {
            m_elements[m_setCount] = Element(initialElements[m_setCount], m_setCount);
        }
    }

    //#################### PUBLIC METHODS ####################
public:
    /**
    @brief  Adds a single element x (and its associated value) to the disjoint set forest.

    @param[in]  x       The index of the element
    @param[in]  value   The value to initially associate with the element
    @pre
        -   x must not already be in the disjoint set forest
    */
    void add_element(size_t x, const T &value = T()) {
        m_elements.insert(std::make_pair(x, Element(value, x)));
        m_setCount++;
    }

    /**
    @brief  Adds multiple elements (and their associated values) to the disjoint set forest.

    @param[in]  elements    A map from the elements to add to their associated values
    @pre
        -   None of the elements to be added must already be in the disjoint set forest
    */
    void add_elements(const std::unordered_map<size_t, T> &initialElements) {
        for (auto it = initialElements.begin(); it != initialElements.end(); ++it) {
            m_elements.insert(std::make_pair(it->first, Element(it->second, it->first)));
        }
        m_setCount += initialElements.size();
    }

    /**
    @brief  Returns the number of elements in the disjoint set forest.

    @return As described
    */
    size_t element_count() const { return m_elements.size(); }

    /**
    @brief  Finds the index of the root element of the tree containing x in the disjoint set forest.

    @param[in]  x   The element whose set to determine
    @pre
        -   x must be an element in the disjoint set forest
    @throw Exception
        -   If the precondition is violated
    @return As described
    */
    size_t find_set(size_t x) {
        const Element &element = get_element(x);
        size_t parent = element.m_parent;
        if (parent != x) {
            parent = find_set(parent);
        }
        return parent;
    }

    /**
    @brief  Returns the current number of disjoint sets in the forest (i.e. the current number of trees).

    @return As described
    */
    size_t set_count() const { return m_setCount; }

    /**
    @brief  Merges the disjoint sets containing elements x and y.

    If both elements are already in the same disjoint set, this is a no-op.

    @param[in]  x   The first element
    @param[in]  y   The second element
    @pre
        -   Both x and y must be elements in the disjoint set forest
    @throw Exception
        -   If the precondition is violated
    */
    void union_sets(size_t x, size_t y) {
        size_t setX = find_set(x);
        size_t setY = find_set(y);
        if (setX != setY) link(setX, setY);
    }

    /**
    @brief  Returns the value associated with element x.

    @param[in]  x   The element whose value to return
    @pre
        -   x must be an element in the disjoint set forest
    @throw Exception
        -   If the precondition is violated
    @return As described
    */
    T &value_of(size_t x) { return get_element(x).m_value; }

    /**
    @brief  Returns the value associated with element x.

    @param[in]  x   The element whose value to return
    @pre
        -   x must be an element in the disjoint set forest
    @throw Exception
        -   If the precondition is violated
    @return As described
    */
    const T &value_of(size_t x) const { return get_element(x).m_value; }

    //#################### PRIVATE METHODS ####################
private:
    Element& get_element(size_t x) {
        auto it = m_elements.find(x);
        if (it != m_elements.end())
            return it->second;

        throw std::runtime_error("No such element");
    }

    void link(size_t x, size_t y) {
        Element &elementX = get_element(x);
        Element &elementY = get_element(y);
        size_t& rankX = elementX.m_rank;
        size_t& rankY = elementY.m_rank;
        if (rankX > rankY) {
            elementY.m_parent = x;
        } else {
            elementX.m_parent = y;
            if (rankX == rankY) ++rankY;
        }
        --m_setCount;
    }
};

#endif