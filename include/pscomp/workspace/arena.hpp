// Bump arena allocator with snapshot/restore.

#ifndef PSCOMP_WORKSPACE_ARENA_HPP
#define PSCOMP_WORKSPACE_ARENA_HPP

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

namespace pscomp {

class Arena {
public:
    explicit Arena(std::size_t bytes)
        : storage_(bytes), top_(storage_.data()), end_(storage_.data() + bytes) {}

    template <class T>
    T* allocate(std::size_t count) {
        constexpr std::size_t align = alignof(T) > 32 ? alignof(T) : 32;
        std::uintptr_t cur = reinterpret_cast<std::uintptr_t>(top_);
        cur = (cur + align - 1) & ~(static_cast<std::uintptr_t>(align) - 1);
        std::uintptr_t next = cur + sizeof(T) * count;
        if (next > reinterpret_cast<std::uintptr_t>(end_)) {
            throw std::bad_alloc{};
        }
        top_ = reinterpret_cast<std::byte*>(next);
        return std::launder(reinterpret_cast<T*>(cur));
    }

    void reset() noexcept { top_ = storage_.data(); }

    std::size_t bytes_used() const noexcept {
        return static_cast<std::size_t>(top_ - storage_.data());
    }

    std::size_t capacity() const noexcept { return storage_.size(); }

    struct Marker { std::byte* top; };
    Marker mark() const noexcept { return {top_}; }
    void   release(Marker m) noexcept { assert(m.top <= top_); top_ = m.top; }

private:
    std::vector<std::byte> storage_;
    std::byte*             top_;
    std::byte*             end_;
};

}  // namespace pscomp

#endif
