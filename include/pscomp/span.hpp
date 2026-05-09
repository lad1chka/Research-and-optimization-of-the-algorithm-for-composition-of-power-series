// Minimal C++17 substitute for std::span<T>.

#ifndef PSCOMP_SPAN_HPP
#define PSCOMP_SPAN_HPP

#include <cstddef>
#include <type_traits>
#include <vector>

namespace pscomp {

template <class T>
class span {
public:
    using element_type    = T;
    using value_type      = std::remove_cv_t<T>;
    using size_type       = std::size_t;
    using pointer         = T*;
    using reference       = T&;
    using iterator        = T*;
    using const_iterator  = const T*;

    constexpr span() noexcept = default;
    constexpr span(T* data, size_type size) noexcept : data_(data), size_(size) {}

    template <class U, class A,
              class = std::enable_if_t<std::is_same_v<value_type, U>>>
    constexpr span(std::vector<U, A>& v) noexcept
        : data_(v.data()), size_(v.size()) {}

    template <class U, class A,
              class = std::enable_if_t<std::is_const_v<T> &&
                                       std::is_same_v<value_type, U>>>
    constexpr span(const std::vector<U, A>& v) noexcept
        : data_(v.data()), size_(v.size()) {}

    template <class U,
              class = std::enable_if_t<std::is_const_v<T> &&
                                       std::is_same_v<U, value_type>>>
    constexpr span(span<U> other) noexcept
        : data_(other.data()), size_(other.size()) {}

    constexpr pointer   data()  const noexcept { return data_; }
    constexpr size_type size()  const noexcept { return size_; }
    constexpr bool      empty() const noexcept { return size_ == 0; }
    constexpr reference operator[](size_type i) const noexcept { return data_[i]; }
    constexpr iterator  begin() const noexcept { return data_; }
    constexpr iterator  end()   const noexcept { return data_ + size_; }

    constexpr span first(size_type n) const noexcept { return {data_, n}; }
    constexpr span last(size_type n)  const noexcept { return {data_ + size_ - n, n}; }
    constexpr span subspan(size_type off,
                           size_type cnt = static_cast<size_type>(-1)) const noexcept {
        return cnt == static_cast<size_type>(-1)
                   ? span(data_ + off, size_ - off)
                   : span(data_ + off, cnt);
    }

private:
    T*        data_ = nullptr;
    size_type size_ = 0;
};

}  // namespace pscomp

#endif
