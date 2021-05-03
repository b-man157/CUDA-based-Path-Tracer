#ifndef VARIANT_HPP
#define VARIANT_HPP

#include <type_traits>

// Source: https://stackoverflow.com/a/26169248
// Index_v<T, Ts...> returns the index of T in the parameter pack Ts.

template <typename T, typename ...Ts>
struct Index;

template <typename T, typename ...Ts>
struct Index<T, T, Ts...> : std::integral_constant<std::size_t, 0> {};

template <typename T, typename U, typename ...Ts>
struct Index<T, U, Ts...> : std::integral_constant<std::size_t, 1 + Index<T, Ts...>::value> {};

template <typename T, typename ...Ts>
constexpr std::size_t Index_v = Index<T, Ts...>::value;

// Source: S. Zellman and U. Lang, 2017. C++ Compile Time Polymorphism for Ray Tracing. 131 p.
// Here, Variant uses variadic templates to implement Compile-Time-Polymorphism.

template <typename ...Ts>
union VariantStorage {};

template <typename T, typename ...Ts>
union VariantStorage<T, Ts...> {
    T element;
    VariantStorage<Ts...> nextElements;
};

template <typename ...Ts>
struct Variant {
    VariantStorage<Ts...> storage;
    int type_id;

    template <typename T>
    T* as() {
        if (type_id == Index_v<T, Ts...>)
            return reinterpret_cast<T *>(&storage);
        else
            return nullptr;
    }
};

// applyVisitor() traverses the parameter pack, and calls the appropriate Visitor::operator()
// overload.

class Visitor {};

template <typename T, typename ...Ts>
auto applyVisitor(Visitor visitor, Variant<Ts...> *var) {
    if (!var->template as<T>()) {
        applyVisitor<Ts...>(visitor, var);
    }
    return visitor(var->template as<T>());
}

#endif
