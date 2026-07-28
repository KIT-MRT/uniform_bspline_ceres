#pragma once

#include <array>
#include <tuple>
#include <utility>

#include <Eigen/Core>

namespace test_util {
namespace internal {
template <std::size_t CurDim_, std::size_t N_>
struct LinspaceImpl {
    template <typename Func>
    static void apply(const std::array<int, N_>& counts, Eigen::Matrix<double, N_, 1>& val, Func func) {
        for (int i = 0; i < counts[CurDim_]; ++i) {
            val[CurDim_] = double(i) / (counts[CurDim_] - 1);
            LinspaceImpl<CurDim_ + 1, N_>::apply(counts, val, func);
        }
    }
};

template <std::size_t N_>
struct LinspaceImpl<N_, N_> {
    template <typename Func>
    static void apply(const std::array<int, N_>& /*counts*/, Eigen::Matrix<double, N_, 1>& val, Func func) {
        func(val);
    }
};
} // namespace internal

template <std::size_t N, typename Func>
void linspace(const std::array<int, N>& counts, Func func) {
    Eigen::Matrix<double, N, 1> val;
    internal::LinspaceImpl<0, N>::apply(counts, val, func);
}

namespace internal {
template <class F, class Tuple, std::size_t... I>
constexpr decltype(auto) applyImpl(F&& f, Tuple&& t, std::index_sequence<I...> /*indexSequence*/) {
    return f(std::get<I>(std::forward<Tuple>(t))...);
}
} // namespace internal

template <class F, class Tuple>
constexpr decltype(auto) apply(F&& f, Tuple&& t) {
    return internal::applyImpl(std::forward<F>(f),
                               std::forward<Tuple>(t),
                               std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

namespace internal {
template <typename T, size_t N, size_t... Is>
inline auto asTuple(std::array<T, N> const& arr, std::index_sequence<Is...> /*indexSequence*/) {
    return std::make_tuple(arr[Is]...);
}
} // namespace internal

template <typename T, size_t N>
inline auto asTuple(std::array<T, N> const& arr) {
    return internal::asTuple(arr, std::make_index_sequence<N>{});
}

} // namespace test_util