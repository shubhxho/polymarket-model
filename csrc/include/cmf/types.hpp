#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace cmf {

inline constexpr int kHistory = 64;
inline constexpr int kFastDim = 24;
inline constexpr int kSlowDim = 16;
inline constexpr int kPosDim = 8;
inline constexpr int kLagDim = 8;
inline constexpr int kBookLevels = 10;
inline constexpr int kHawkesWindow = 32;
inline constexpr int kVpinBuckets = 16;

inline constexpr int kRawFastDim = 16;
inline constexpr int kRawSlowDim = 12;

inline constexpr float kEps = 1e-8f;

inline float clampf(float x, float lo = -1.0f, float hi = 1.0f) {
    return std::max(lo, std::min(hi, x));
}

inline float tanhf_fast(float x) {
    x = clampf(x, -8.0f, 8.0f);
    return std::tanh(x);
}

inline float safe_div(float a, float b) {
    return a / (std::fabs(b) + kEps);
}

struct Level {
    float price = 0.0f;
    float size = 0.0f;
};

struct Book {
    std::array<Level, kBookLevels> bids{};
    std::array<Level, kBookLevels> asks{};
    int n_bids = 0;
    int n_asks = 0;

    [[nodiscard]] float best_bid() const { return n_bids > 0 ? bids[0].price : 0.0f; }
    [[nodiscard]] float best_ask() const { return n_asks > 0 ? asks[0].price : 0.0f; }

    [[nodiscard]] float mid() const {
        if (n_bids == 0 || n_asks == 0) {
            return 0.0f;
        }
        return 0.5f * (bids[0].price + asks[0].price);
    }

    [[nodiscard]] float microprice() const {
        if (n_bids == 0 || n_asks == 0) {
            return mid();
        }
        const float bs = bids[0].size;
        const float as = asks[0].size;
        const float den = bs + as;
        if (den <= kEps) {
            return mid();
        }
        return (asks[0].price * bs + bids[0].price * as) / den;
    }

    [[nodiscard]] float spread() const {
        if (n_bids == 0 || n_asks == 0) {
            return 0.0f;
        }
        return std::max(0.0f, asks[0].price - bids[0].price);
    }

    [[nodiscard]] float depth(bool is_bid, int levels) const {
        const int n = is_bid ? n_bids : n_asks;
        const auto& side = is_bid ? bids : asks;
        const int lim = std::min(n, levels);
        float s = 0.0f;
        for (int i = 0; i < lim; ++i) {
            s += side[i].size;
        }
        return s;
    }

    [[nodiscard]] float imbalance(int levels) const {
        const float b = depth(true, levels);
        const float a = depth(false, levels);
        return safe_div(b - a, b + a);
    }

    [[nodiscard]] float slope(bool is_bid, int levels) const {
        const int n = is_bid ? n_bids : n_asks;
        const auto& side = is_bid ? bids : asks;
        const int lim = std::min(n, levels);
        if (lim < 2) {
            return 0.0f;
        }
        const float m = mid();
        float num = 0.0f;
        float den = 0.0f;
        for (int i = 0; i < lim; ++i) {
            const float dx = std::fabs(side[i].price - m);
            num += dx * side[i].size;
            den += dx * dx;
        }
        return safe_div(num, den);
    }
};

struct Trade {
    double ts = 0.0;
    float price = 0.0f;
    float qty = 0.0f;
    float sign = 0.0f;  // +1 buy, -1 sell
};

struct Position {
    float has = 0.0f;
    float side = 0.0f;          // +1 UP, -1 DOWN
    float pnl = 0.0f;
    float entry = 0.0f;
    float time_in = 0.0f;
    float time_remaining = 1.0f;
    float shares = 0.0f;
    float inventory_risk = 0.0f;
};

struct FeatureFrame {
    std::array<float, kFastDim> fast{};
    std::array<float, kSlowDim> slow{};
    std::array<float, kPosDim> pos{};
    std::array<float, kLagDim> lag{};
};

// Raw tick columns consumed by the batch featurizer.
// Fast: ts, mid, bid, ask, bid_sz, ask_sz, trade_qty, trade_sign,
//       volume, n_trades, funding, oi, liq_long, liq_short, basis, jump
// Slow: ts, mid, bid, ask, bid_sz, ask_sz, trade_qty, trade_sign,
//       depth_bid, depth_ask, time_remaining, stale_sec
inline constexpr int kFastRawTs = 0;
inline constexpr int kFastRawMid = 1;
inline constexpr int kFastRawBid = 2;
inline constexpr int kFastRawAsk = 3;
inline constexpr int kFastRawBidSz = 4;
inline constexpr int kFastRawAskSz = 5;
inline constexpr int kFastRawQty = 6;
inline constexpr int kFastRawSign = 7;
inline constexpr int kFastRawVol = 8;
inline constexpr int kFastRawNTrades = 9;
inline constexpr int kFastRawFunding = 10;
inline constexpr int kFastRawOi = 11;
inline constexpr int kFastRawLiqLong = 12;
inline constexpr int kFastRawLiqShort = 13;
inline constexpr int kFastRawBasis = 14;
inline constexpr int kFastRawJump = 15;

inline constexpr int kSlowRawTs = 0;
inline constexpr int kSlowRawMid = 1;
inline constexpr int kSlowRawBid = 2;
inline constexpr int kSlowRawAsk = 3;
inline constexpr int kSlowRawBidSz = 4;
inline constexpr int kSlowRawAskSz = 5;
inline constexpr int kSlowRawQty = 6;
inline constexpr int kSlowRawSign = 7;
inline constexpr int kSlowRawDepthBid = 8;
inline constexpr int kSlowRawDepthAsk = 9;
inline constexpr int kSlowRawTte = 10;
inline constexpr int kSlowRawStale = 11;

}  // namespace cmf
