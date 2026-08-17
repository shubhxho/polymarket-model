#pragma once

#include "cmf/ring.hpp"
#include "cmf/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>

namespace cmf {

struct VenueState {
    Book book{};
    bool have_prev_book = false;
    double last_ts = 0.0;
    float last_mid = 0.0f;
    float last_micro = 0.0f;

    Ring<float, kHistory> mids;
    Ring<float, kHistory> rets;
    Ring<float, kHistory> signed_vol;
    Ring<Trade, kHawkesWindow> trades;

    float cvd = 0.0f;
    float prev_cvd = 0.0f;
    float ofi_l1 = 0.0f;
    float ofi_l5 = 0.0f;
    float hawkes_buy = 0.0f;
    float hawkes_sell = 0.0f;
    float vpin = 0.0f;
    float kyle_lambda = 0.0f;
    float large_flag = 0.0f;
    float trade_notional_ema = 1.0f;

    float bucket_buy = 0.0f;
    float bucket_sell = 0.0f;
    float bucket_vol = 0.0f;
    float bucket_target = 0.0f;
    Ring<float, kVpinBuckets> vpin_imb;

    void reset() { *this = VenueState{}; }
};

class FusionEngine {
public:
    FusionEngine() = default;

    void reset() {
        fast_.reset();
        slow_.reset();
        pos_ = {};
        extras_ = {};
        slow_stale_ = 0.0f;
    }

    void set_position(const Position& p) { pos_ = p; }

    void set_fast_extras(float funding, float oi_chg, float liq, float basis) {
        extras_[0] = tanhf_fast(funding * 200.0f);
        extras_[1] = tanhf_fast(oi_chg * 20.0f);
        extras_[2] = tanhf_fast(liq);
        extras_[3] = tanhf_fast(basis * 200.0f);
    }

    void set_slow_stale(float stale_sec) { slow_stale_ = stale_sec; }

    void push_fast_book(double ts, const Book& book) { apply_book(fast_, ts, book); }
    void push_slow_book(double ts, const Book& book) { apply_book(slow_, ts, book); }

    void push_fast_trade(double ts, float price, float qty, float sign) {
        apply_trade(fast_, ts, price, qty, sign);
    }
    void push_slow_trade(double ts, float price, float qty, float sign) {
        apply_trade(slow_, ts, price, qty, sign);
    }

    void ingest_fast_raw(std::span<const float, kRawFastDim> r) {
        Book b{};
        b.n_bids = 1;
        b.n_asks = 1;
        b.bids[0] = {r[kFastRawBid], std::max(r[kFastRawBidSz], 1e-4f)};
        b.asks[0] = {r[kFastRawAsk], std::max(r[kFastRawAskSz], 1e-4f)};
        if (b.bids[0].price <= 0.0f && r[kFastRawMid] > 0.0f) {
            b.bids[0].price = r[kFastRawMid] * 0.9995f;
            b.asks[0].price = r[kFastRawMid] * 1.0005f;
        }
        apply_book(fast_, static_cast<double>(r[kFastRawTs]), b);
        if (std::fabs(r[kFastRawQty]) > 0.0f) {
            apply_trade(fast_, static_cast<double>(r[kFastRawTs]),
                        r[kFastRawMid] > 0.0f ? r[kFastRawMid] : b.mid(),
                        r[kFastRawQty], r[kFastRawSign]);
        }
        set_fast_extras(
            r[kFastRawFunding],
            r[kFastRawOi],
            safe_div(r[kFastRawLiqLong] - r[kFastRawLiqShort],
                     r[kFastRawLiqLong] + r[kFastRawLiqShort] + 1.0f),
            r[kFastRawBasis]
        );
    }

    void ingest_slow_raw(std::span<const float, kRawSlowDim> r) {
        Book b{};
        b.n_bids = 2;
        b.n_asks = 2;
        const float mid = r[kSlowRawMid];
        const float bid = r[kSlowRawBid] > 0.0f ? r[kSlowRawBid] : clampf(mid - 0.01f, 0.01f, 0.99f);
        const float ask = r[kSlowRawAsk] > 0.0f ? r[kSlowRawAsk] : clampf(mid + 0.01f, 0.01f, 0.99f);
        b.bids[0] = {bid, std::max(r[kSlowRawBidSz], 1.0f)};
        b.asks[0] = {ask, std::max(r[kSlowRawAskSz], 1.0f)};
        b.bids[1] = {clampf(bid - 0.01f, 0.01f, 0.99f), std::max(r[kSlowRawDepthBid], 1.0f)};
        b.asks[1] = {clampf(ask + 0.01f, 0.01f, 0.99f), std::max(r[kSlowRawDepthAsk], 1.0f)};
        apply_book(slow_, static_cast<double>(r[kSlowRawTs]), b);
        if (std::fabs(r[kSlowRawQty]) > 0.0f) {
            apply_trade(slow_, static_cast<double>(r[kSlowRawTs]), mid, r[kSlowRawQty], r[kSlowRawSign]);
        }
        set_slow_stale(r[kSlowRawStale]);
        pos_.time_remaining = r[kSlowRawTte];
    }

    [[nodiscard]] FeatureFrame snapshot() const {
        FeatureFrame out{};
        fill_fast(out.fast);
        fill_slow(out.slow);
        fill_pos(out.pos);
        fill_lag(out.lag);
        return out;
    }

    void featurize(
        std::span<const float> fast_raw,
        std::span<const float> slow_raw,
        int ticks,
        std::span<float> fast_out,
        std::span<float> slow_out,
        std::span<float> lag_out
    ) {
        reset();
        if (ticks <= 0) {
            return;
        }
        for (int t = 0; t < ticks; ++t) {
            ingest_fast_raw(std::span<const float, kRawFastDim>(
                fast_raw.data() + static_cast<std::size_t>(t) * kRawFastDim, kRawFastDim));
            ingest_slow_raw(std::span<const float, kRawSlowDim>(
                slow_raw.data() + static_cast<std::size_t>(t) * kRawSlowDim, kRawSlowDim));
            const FeatureFrame frame = snapshot();
            std::copy(frame.fast.begin(), frame.fast.end(),
                      fast_out.data() + static_cast<std::size_t>(t) * kFastDim);
            std::copy(frame.slow.begin(), frame.slow.end(),
                      slow_out.data() + static_cast<std::size_t>(t) * kSlowDim);
            std::copy(frame.lag.begin(), frame.lag.end(),
                      lag_out.data() + static_cast<std::size_t>(t) * kLagDim);
        }
    }

private:
    VenueState fast_{};
    VenueState slow_{};
    Position pos_{};
    std::array<float, 4> extras_{};
    float slow_stale_ = 0.0f;

    static float ofi_side(const Book& prev, const Book& cur, bool is_bid, int levels) {
        const int pn = is_bid ? prev.n_bids : prev.n_asks;
        const int cn = is_bid ? cur.n_bids : cur.n_asks;
        const auto& ps = is_bid ? prev.bids : prev.asks;
        const auto& cs = is_bid ? cur.bids : cur.asks;
        const int lim = std::min({levels, pn, cn});
        float acc = 0.0f;
        for (int i = 0; i < lim; ++i) {
            if (cs[i].price > ps[i].price) {
                acc += is_bid ? cs[i].size : -ps[i].size;
            } else if (cs[i].price < ps[i].price) {
                acc += is_bid ? -ps[i].size : cs[i].size;
            } else {
                acc += (is_bid ? 1.0f : -1.0f) * (cs[i].size - ps[i].size);
            }
        }
        return acc;
    }

    static void apply_book(VenueState& v, double ts, const Book& book) {
        if (v.have_prev_book) {
            v.ofi_l1 = ofi_side(v.book, book, true, 1) + ofi_side(v.book, book, false, 1);
            v.ofi_l5 = ofi_side(v.book, book, true, 5) + ofi_side(v.book, book, false, 5);
        }
        v.have_prev_book = v.book.n_bids > 0 || v.book.n_asks > 0;
        v.book = book;
        const float mid = book.mid();
        if (v.last_mid > kEps && mid > kEps) {
            v.rets.push((mid - v.last_mid) / v.last_mid);
        } else if (mid > kEps) {
            v.rets.push(0.0f);
        }
        if (mid > kEps) {
            v.mids.push(mid);
            v.last_mid = mid;
            v.last_micro = book.microprice();
        }
        v.last_ts = ts;
        v.large_flag *= 0.85f;
    }

    static void apply_trade(VenueState& v, double ts, float price, float qty, float sign) {
        const float notional = std::fabs(price * qty);
        v.trade_notional_ema = 0.95f * v.trade_notional_ema + 0.05f * std::max(notional, 1e-4f);
        if (notional > 3.0f * v.trade_notional_ema) {
            v.large_flag = 1.0f;
        }
        v.prev_cvd = v.cvd;
        v.cvd += sign * notional;
        v.signed_vol.push(sign * notional);
        v.trades.push(Trade{ts, price, qty, sign});

        constexpr float kBeta = 1.6f;
        constexpr float kAlpha = 0.55f;
        if (v.last_ts > 0.0) {
            const float dt = static_cast<float>(std::max(0.0, ts - v.last_ts));
            const float decay = std::exp(-kBeta * dt);
            v.hawkes_buy *= decay;
            v.hawkes_sell *= decay;
        }
        if (sign > 0.0f) {
            v.hawkes_buy += kAlpha;
        } else if (sign < 0.0f) {
            v.hawkes_sell += kAlpha;
        }

        if (v.bucket_target <= kEps) {
            v.bucket_target = std::max(notional * 8.0f, 1.0f);
        }
        if (sign > 0.0f) {
            v.bucket_buy += notional;
        } else {
            v.bucket_sell += notional;
        }
        v.bucket_vol += notional;
        if (v.bucket_vol >= v.bucket_target) {
            v.vpin_imb.push(safe_div(std::fabs(v.bucket_buy - v.bucket_sell), v.bucket_vol));
            v.bucket_buy = 0.0f;
            v.bucket_sell = 0.0f;
            v.bucket_vol = 0.0f;
            float acc = 0.0f;
            for (std::size_t i = 0; i < v.vpin_imb.size(); ++i) {
                acc += v.vpin_imb[i];
            }
            v.vpin = v.vpin_imb.empty() ? 0.0f : acc / static_cast<float>(v.vpin_imb.size());
        }

        if (v.rets.size() >= 8 && v.signed_vol.size() >= 8) {
            const std::size_t n = std::min(v.rets.size(), v.signed_vol.size());
            const std::size_t w = std::min<std::size_t>(n, 16);
            float num = 0.0f;
            float den = 0.0f;
            for (std::size_t i = n - w; i < n; ++i) {
                const float sv = v.signed_vol[i];
                num += v.rets[i] * sv;
                den += sv * sv;
            }
            v.kyle_lambda = safe_div(num, den);
        }
        v.last_ts = ts;
    }

    static float window_return(const VenueState& v, int n) {
        if (v.mids.size() < 2) {
            return 0.0f;
        }
        const int last = static_cast<int>(v.mids.size()) - 1;
        const int idx = std::max(0, last - n);
        const float a = v.mids[static_cast<std::size_t>(idx)];
        const float b = v.mids[static_cast<std::size_t>(last)];
        return safe_div(b - a, a);
    }

    static float realized_vol(const VenueState& v, int n) {
        if (v.rets.size() < 2) {
            return 0.0f;
        }
        const int last = static_cast<int>(v.rets.size());
        const int start = std::max(0, last - n);
        float s2 = 0.0f;
        int c = 0;
        for (int i = start; i < last; ++i) {
            const float r = v.rets[static_cast<std::size_t>(i)];
            s2 += r * r;
            ++c;
        }
        return std::sqrt(s2 / static_cast<float>(std::max(c, 1)));
    }

    static float trade_intensity(const VenueState& v, double now, float horizon) {
        if (v.trades.empty()) {
            return 0.0f;
        }
        int c = 0;
        v.trades.for_each([&](const Trade& tr) {
            if (now - tr.ts <= static_cast<double>(horizon)) {
                ++c;
            }
        });
        return static_cast<float>(c) / std::max(horizon, 1.0f);
    }

    void fill_fast(std::array<float, kFastDim>& o) const {
        const auto& v = fast_;
        const float rv5 = realized_vol(v, 5);
        const float rv20 = realized_vol(v, 20);
        const float ofi_scale = 1.0f / std::max(v.trade_notional_ema, 1.0f);
        const float cvd_acc = (v.cvd - v.prev_cvd) * ofi_scale;
        const float flow = (v.hawkes_buy - v.hawkes_sell) / (v.hawkes_buy + v.hawkes_sell + 1.0f);
        o[0] = tanhf_fast(window_return(v, 1) * 80.0f);
        o[1] = tanhf_fast(window_return(v, 5) * 40.0f);
        o[2] = tanhf_fast(window_return(v, 15) * 25.0f);
        o[3] = tanhf_fast(window_return(v, 30) * 18.0f);
        o[4] = tanhf_fast(window_return(v, 60) * 12.0f);
        o[5] = tanhf_fast(std::log1p(std::max(v.trade_notional_ema, 0.0f)) / 8.0f);
        o[6] = tanhf_fast(rv20 * 80.0f);
        o[7] = tanhf_fast(safe_div(rv5, rv20 + kEps) - 1.0f);
        o[8] = tanhf_fast(v.ofi_l1 * ofi_scale);
        o[9] = tanhf_fast(v.ofi_l5 * ofi_scale * 0.4f);
        o[10] = tanhf_fast(flow);
        o[11] = tanhf_fast(cvd_acc);
        o[12] = tanhf_fast(v.kyle_lambda * 50.0f);
        o[13] = tanhf_fast(v.hawkes_buy + v.hawkes_sell);
        o[14] = clampf(v.vpin, 0.0f, 1.0f);
        o[15] = clampf(v.large_flag, 0.0f, 1.0f);
        o[16] = tanhf_fast(trade_intensity(v, v.last_ts, 8.0f) / 6.0f);
        o[17] = tanhf_fast(v.book.slope(true, 5) - v.book.slope(false, 5));
        o[18] = tanhf_fast(safe_div(v.last_micro - v.last_mid, v.last_mid + kEps) * 200.0f);
        o[19] = extras_[0];
        o[20] = extras_[1];
        o[21] = extras_[2];
        o[22] = extras_[3];
        const float jump_dir = v.rets.empty() ? 0.0f : (v.rets.newest() > 0.0f ? 1.0f : -1.0f);
        o[23] = clampf(v.large_flag * jump_dir, -1.0f, 1.0f);
    }

    void fill_slow(std::array<float, kSlowDim>& o) const {
        const auto& v = slow_;
        const float mid = v.last_mid;
        const float spr = v.book.spread();
        const float vel = window_return(v, 3);
        float vel_prev = 0.0f;
        if (v.mids.size() >= 6) {
            const auto n = v.mids.size();
            const float a = v.mids[n - 6];
            const float b = v.mids[n - 3];
            vel_prev = safe_div(b - a, a);
        }
        o[0] = clampf(2.0f * mid - 1.0f);
        o[1] = clampf(2.0f * v.last_micro - 1.0f);
        o[2] = tanhf_fast(safe_div(spr, std::max(mid, 0.05f)) * 20.0f);
        o[3] = clampf(v.book.imbalance(1));
        o[4] = clampf(v.book.imbalance(5));
        o[5] = clampf(v.book.imbalance(10));
        o[6] = tanhf_fast(v.book.slope(true, 5) - v.book.slope(false, 5));
        o[7] = tanhf_fast(vel * 25.0f);
        o[8] = tanhf_fast((vel - vel_prev) * 25.0f);
        o[9] = tanhf_fast(trade_intensity(v, v.last_ts, 8.0f) / 4.0f);
        o[10] = clampf(v.large_flag);
        o[11] = tanhf_fast(std::log1p(v.book.depth(true, 5)) / 6.0f);
        o[12] = tanhf_fast(std::log1p(v.book.depth(false, 5)) / 6.0f);
        o[13] = clampf(pos_.time_remaining, 0.0f, 1.0f);
        o[14] = clampf(2.0f * (mid - 0.5f), -1.0f, 1.0f);
        o[15] = tanhf_fast(slow_stale_ / 8.0f);
    }

    void fill_pos(std::array<float, kPosDim>& o) const {
        o[0] = pos_.has;
        o[1] = clampf(pos_.side);
        o[2] = tanhf_fast(pos_.pnl / 25.0f);
        o[3] = clampf(2.0f * pos_.entry - 1.0f);
        o[4] = clampf(pos_.time_in);
        o[5] = clampf(pos_.time_remaining);
        o[6] = tanhf_fast(pos_.shares / 20.0f);
        o[7] = tanhf_fast(pos_.inventory_risk);
    }

    void fill_lag(std::array<float, kLagDim>& o) const {
        constexpr int lags[6] = {0, 1, 2, 4, 8, 16};
        float best_c = -2.0f;
        int best_k = 0;
        float c0 = 0.0f;
        const auto n = std::min(fast_.rets.size(), slow_.rets.size());
        for (int li = 0; li < 6; ++li) {
            const int k = lags[li];
            const float c = lagged_corr(k, n);
            o[static_cast<std::size_t>(li)] = clampf(c);
            if (k == 0) {
                c0 = c;
            }
            if (c > best_c) {
                best_c = c;
                best_k = k;
            }
        }
        o[6] = clampf(best_c - c0);
        o[7] = clampf(static_cast<float>(best_k) / 16.0f);
    }

    [[nodiscard]] float lagged_corr(int k, std::size_t n) const {
        if (n < static_cast<std::size_t>(k) + 8) {
            return 0.0f;
        }
        const std::size_t m = n - static_cast<std::size_t>(k);
        float sx = 0.0f, sy = 0.0f, sxx = 0.0f, syy = 0.0f, sxy = 0.0f;
        const std::size_t start = (m > 48) ? m - 48 : 0;
        int c = 0;
        for (std::size_t i = start; i < m; ++i) {
            const float x = fast_.rets[i];
            const float y = slow_.rets[i + static_cast<std::size_t>(k)];
            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
            ++c;
        }
        if (c < 4) {
            return 0.0f;
        }
        const float inv = 1.0f / static_cast<float>(c);
        const float mx = sx * inv;
        const float my = sy * inv;
        const float cov = sxy * inv - mx * my;
        const float vx = sxx * inv - mx * mx;
        const float vy = syy * inv - my * my;
        return safe_div(cov, std::sqrt(std::max(vx, 0.0f) * std::max(vy, 0.0f)));
    }
};

inline void featurize_episode(
    std::span<const float> fast_raw,
    std::span<const float> slow_raw,
    int ticks,
    std::span<float> fast_out,
    std::span<float> slow_out,
    std::span<float> lag_out
) {
    FusionEngine eng;
    eng.featurize(fast_raw, slow_raw, ticks, fast_out, slow_out, lag_out);
}

}  // namespace cmf
