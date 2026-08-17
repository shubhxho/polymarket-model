#pragma once

#include "cmf/types.hpp"

#include <cstddef>

namespace cmf {

template <typename T, std::size_t N>
class Ring {
public:
    static_assert(N > 0);

    void clear() {
        head_ = 0;
        count_ = 0;
    }

    void push(const T& value) {
        buf_[head_] = value;
        head_ = (head_ + 1) % N;
        if (count_ < N) {
            ++count_;
        }
    }

    [[nodiscard]] std::size_t size() const { return count_; }
    [[nodiscard]] bool empty() const { return count_ == 0; }
    [[nodiscard]] bool full() const { return count_ == N; }
    [[nodiscard]] static constexpr std::size_t capacity() { return N; }

    // 0 = oldest, size()-1 = newest
    [[nodiscard]] const T& operator[](std::size_t i) const {
        const std::size_t start = (head_ + N - count_) % N;
        return buf_[(start + i) % N];
    }

    [[nodiscard]] T& operator[](std::size_t i) {
        const std::size_t start = (head_ + N - count_) % N;
        return buf_[(start + i) % N];
    }

    [[nodiscard]] const T& newest() const { return (*this)[count_ - 1]; }
    [[nodiscard]] const T& oldest() const { return (*this)[0]; }

    template <typename F>
    void for_each(F&& fn) const {
        for (std::size_t i = 0; i < count_; ++i) {
            fn((*this)[i]);
        }
    }

private:
    std::array<T, N> buf_{};
    std::size_t head_ = 0;
    std::size_t count_ = 0;
};

}  // namespace cmf
