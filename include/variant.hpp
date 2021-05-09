/* Source of idea: C++ Compile Time Polymorphism for Ray Tracing - S. Zellman and U. Lang, 2017.
 * Code partially taken (with modification) from:
 *        https://github.com/szellmann/visionaray/blob/master/include/visionaray/variant.h */

#ifndef VARIANT_HPP
#define VARIANT_HPP

#include <type_traits>

#ifdef __CUDACC__
    #define __HD__ __host__ __device__
#else
    #define __device__
    #define __HD__
#endif

// std::size_t indexOf<T, Ts...> returns the index of T in the parameter pack Ts.

template <typename ...Ts>
struct indexOf_impl;

template <typename T, typename ...Ts>
struct indexOf_impl<T, T, Ts...> : std::integral_constant<std::size_t, 1> {};

template <typename T, typename U, typename ...Ts>
struct indexOf_impl<T, U, Ts...> :
    std::integral_constant<std::size_t, 1 + indexOf_impl<T, Ts...>::value> {};

template <typename T, typename ...Ts>
constexpr std::size_t indexOf = indexOf_impl<T, Ts...>::value;

// typeAt() returns the Ith typename in the parameter pack.

template <unsigned I, typename ...Ts>
struct typeAt_impl;

template <typename T, typename ...Ts>
struct typeAt_impl<1, T, Ts...> { using type = T; };

template <unsigned I, typename T, typename ...Ts>
struct typeAt_impl<I, T, Ts...> : typeAt_impl<I - 1, Ts...> {};

template <unsigned I, typename ...Ts>
using typeAt = typename typeAt_impl<I, Ts...>::type;

// typedIndex<I> returns the corresponding type for I, needed in templates.

template <unsigned I>
using typedIndex = std::integral_constant<unsigned, I>;

// union VariantStorage recursively creates and stores the values for each type in the parameter
// pack.

template <typename ...Ts>
union VariantStorage {};

template <typename T, typename ...Ts>
union VariantStorage<T, Ts...> {
    T element;
    VariantStorage<Ts...> nextElements;

    __HD__ VariantStorage() {}

    __HD__ T &get(typedIndex<1>) { return element; }
    __HD__ const T &get(typedIndex<1>) const { return element; }

    template <unsigned I, std::enable_if_t<(I > 1), bool> = true>
    __HD__ typeAt<I - 1, Ts...> &get(typedIndex<I>) {
        return nextElements.get(typedIndex<I - 1>{});
    }

    template <unsigned I, std::enable_if_t<(I > 1), bool> = true>
    __HD__ typeAt<I - 1, Ts...> const &get(typedIndex<I>) const {
        return nextElements.get(typedIndex<I - 1>{});
    }
};

// class Variant<Ts...> implements the required polymorphism functionality by inferring types at
// compile-time.

template <typename ...Ts>
class Variant {
    public:
        Variant() = default;

        template <typename T>
        __HD__ Variant(const T &value) : type_index_(indexOf<T, Ts...>) {
            storage_.get(typedIndex<indexOf<T, Ts...>>()) = value;
        }

        template <typename T>
        __HD__ Variant &operator=(const T &value) {
            type_index_ = indexOf<T, Ts...>;
            storage_.get(typedIndex<indexOf<T, Ts...>>()) = value;
        }

        template <typename T>
        __HD__ const T* as() const {
            return type_index_ == indexOf<T, Ts...>
                ? &storage_.get(typedIndex<indexOf<T, Ts...>>())
                : nullptr;
        }

    private:
        VariantStorage<Ts...> storage_;
        std::size_t type_index_;
};

// Visitor::return_type applyVisitor(Visitor, Variant) applies Visitor on the Variant.

template <unsigned I, typename ...Ts>
struct applyVisitor_impl;

template <unsigned I, typename T, typename ...Ts>
struct applyVisitor_impl<I, T, Ts...> {
    template <typename Visitor, typename Variant>
    __device__ typename Visitor::return_type operator()(
            Visitor &visitor, const Variant &var) const {
        auto ptr = var.template as<T>();
        if (ptr)
            return visitor(*ptr);
        else
            return applyVisitor_impl<I - 1, Ts...>()(visitor, var);
    }
};

template <>
struct applyVisitor_impl<0> {
    template <typename Visitor, typename Variant>
    __device__ typename Visitor::return_type operator()(
            Visitor &visitor, const Variant &var) {
        return typename Visitor::return_type();
    }
};

template <typename Visitor, typename ...Ts>
__HD__ typename Visitor::return_type applyVisitor(
        Visitor &visitor, const Variant<Ts...> &var) {
    return applyVisitor_impl<sizeof...(Ts), Ts...>()(visitor, var);
}

#ifndef __CUDACC__
    #undef __device__
#endif
#undef __HD__

#endif
