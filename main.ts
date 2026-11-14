// polymarket-faa-attribution-02 entry: a deterministic, offline quantile forecaster.
//
// For every market of a case it estimates a per-minute volatility with an EWMA of squared
// price changes and turns the case's own recent history into an empirical distribution of
// standardized h-step changes (a "fan"), bucketed by price level and horizon. Each forecast
// is the last price plus that fan's quantile, scaled by the market's current volatility.
// Families that look exhaustive (many markets summing to ~1) get a slow sum-to-1 pull; a
// four-market low-probability family does not.
//
// Standard library only; no network; deterministic bytes.

import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];

export interface ModelOptions {
  halfLife: number;
  fanWindow: number;
  fanStride: number;
  fanBuckets: number;
  fanCap: number;
  fanMinN: number;
  fallbackK: number;
  burnIn: number;
  priceBuckets: boolean;
  shrink: number;
  medianZero: boolean;
  medianShrink: number;
  sumPullRho: number;
}

const DEFAULTS: ModelOptions = {
  halfLife: 90,
  fanWindow: 1440,
  fanStride: 2,
  fanBuckets: 2,
  fanCap: 8,
  fanMinN: 100,
  fallbackK: 2.5,
  burnIn: 0,
  priceBuckets: true,
  shrink: 0.75,
  medianZero: false,
  medianShrink: 0.5,
  sumPullRho: 0.99,
};

const PRICE_BUCKET_COUNT = 3;

interface CaseMarket {
  id: string;
  prices: number[];
  start?: string;
}

interface CaseFile {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

interface ForecastRow {
  id: string;
  q: number[][];
}

const clamp6 = (x: number): number => {
  const c = Math.min(1, Math.max(0, x));
  const r = Math.round(c * 1e6) / 1e6;
  return r === 0 ? 0 : r; // never -0
};

/** EWMA standard deviation of 1-minute price changes, one value per price. */
export function sigmaSeries(prices: number[], halfLife: number): number[] {
  const n = prices.length;
  const out = new Array<number>(n).fill(0);
  if (n < 2) return out;
  const lam = Math.pow(0.5, 1 / halfLife);
  let s = 0;
  let w = 0;
  for (let i = 1; i < n; i++) {
    const a = prices[i];
    const b = prices[i - 1];
    if (!Number.isFinite(a) || !Number.isFinite(b)) {
      out[i] = out[i - 1];
      continue;
    }
    const d = a - b;
    s = lam * s + (1 - lam) * d * d;
    w = lam * w + 1;
    out[i] = Math.sqrt(s / w);
  }
  out[0] = out[1] ?? 0;
  return out;
}

/** Linear-interpolated quantile of an ascending array. */
export function quantileSorted(sorted: number[], tau: number): number {
  const n = sorted.length;
  if (n === 0) return 0;
  const pos = (n - 1) * tau;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function lastFinite(prices: number[], fallback: number): number {
  for (let i = prices.length - 1; i >= 0; i--) if (Number.isFinite(prices[i])) return prices[i];
  return fallback;
}

function firstFinite(prices: number[]): number {
  for (let i = 0; i < prices.length; i++) if (Number.isFinite(prices[i])) return i;
  return prices.length;
}

/** Price-level bucket: sub-cent, single-digit cents, or higher. */
function priceBucket(p: number): number {
  if (!Number.isFinite(p)) return 1;
  if (p < 0.01) return 0;
  if (p < 0.1) return 1;
  return 2;
}

/** Builds the case's standardized-change fan: [horizon bucket][price bucket + pooled]. */
function buildFan(c: CaseFile, series: number[][], opts: ModelOptions): number[][][] {
  const pbCount = opts.priceBuckets ? PRICE_BUCKET_COUNT : 0;
  const fan: number[][][] = Array.from({ length: opts.fanBuckets }, () =>
    Array.from({ length: pbCount + 1 }, () => []),
  );
  const H = c.horizon_minutes;
  for (let mi = 0; mi < c.markets.length; mi++) {
    const prices = c.markets[mi].prices;
    const sig = series[mi];
    const n = prices.length;
    const first = firstFinite(prices);
    for (let h = 1; h <= H; h++) {
      const bucket = Math.min(opts.fanBuckets - 1, Math.floor(((h - 1) * opts.fanBuckets) / H));
      const rootH = Math.sqrt(h);
      const from = Math.max(first + opts.burnIn, n - opts.fanWindow);
      for (let t = from; t < n - h; t += opts.fanStride) {
        const denom = sig[t] * rootH;
        if (!(denom > 1e-12)) continue;
        let r = (prices[t + h] - prices[t]) / denom;
        if (!Number.isFinite(r)) continue;
        if (r > opts.fanCap) r = opts.fanCap;
        else if (r < -opts.fanCap) r = -opts.fanCap;
        const pb = opts.priceBuckets ? priceBucket(prices[t]) : 0;
        fan[bucket][pb].push(r);
        fan[bucket][pbCount].push(r);
      }
    }
  }
  for (const horizon of fan) for (const arr of horizon) arr.sort((a, b) => a - b);
  return fan;
}

/** Forecasts every market of one case. */
export function forecastCase(c: CaseFile, options: Partial<ModelOptions> = {}): { forecasts: ForecastRow[] } {
  const opts: ModelOptions = { ...DEFAULTS, ...options };
  const H = c.horizon_minutes;
  const pbCount = opts.priceBuckets ? PRICE_BUCKET_COUNT : 0;
  const series = c.markets.map((m) => sigmaSeries(m.prices, opts.halfLife));
  const fan = buildFan(c, series, opts);

  const positiveSigmas: number[] = [];
  for (const s of series) {
    const v = s[s.length - 1];
    if (Number.isFinite(v) && v > 0) positiveSigmas.push(v);
  }
  positiveSigmas.sort((a, b) => a - b);
  const fallbackSigma = positiveSigmas.length ? positiveSigmas[Math.floor(positiveSigmas.length / 2)] : 0;

  const lasts = c.markets.map((m) => lastFinite(m.prices, 0.5));
  const sum0 = lasts.reduce((a, b) => a + b, 0);
  const nMarkets = c.markets.length;
  const pull = sum0 > 1e-9 && (
    (nMarkets >= 8 && sum0 >= 0.5 && sum0 <= 1.5) ||
    (sum0 >= 0.85 && sum0 <= 1.15)
  );

  const forecasts: ForecastRow[] = [];
  for (let mi = 0; mi < c.markets.length; mi++) {
    const prices = c.markets[mi].prices;
    const n = prices.length;
    const x = lasts[mi];
    const sigLast = series[mi][n - 1];
    const sigma = Number.isFinite(sigLast) && sigLast > 0 ? sigLast : fallbackSigma;
    const pb = opts.priceBuckets ? priceBucket(x) : 0;
    const rows: number[][] = [];
    for (let h = 1; h <= H; h++) {
      const bucket = Math.min(opts.fanBuckets - 1, Math.floor(((h - 1) * opts.fanBuckets) / H));
      const horizonFan = fan[bucket];
      let arr = horizonFan[pb];
      if (arr.length < opts.fanMinN) arr = horizonFan[pbCount];
      const denom = sigma * Math.sqrt(h);
      const row: number[] = [];
      for (let j = 0; j < 5; j++) {
        let z = Z[j] * opts.fallbackK;
        if (arr.length >= opts.fanMinN) {
          const emp = quantileSorted(arr, QUANTILES[j]);
          z = opts.shrink * emp + (1 - opts.shrink) * Z[j] * opts.fallbackK;
        }
        if (j === 2) {
          if (opts.medianZero) z = 0;
          else if (arr.length >= opts.fanMinN) z = opts.medianShrink * quantileSorted(arr, QUANTILES[2]);
        }
        let q = x + z * denom;
        if (pull) {
          const sh = 1 + (sum0 - 1) * Math.pow(opts.sumPullRho, h);
          q *= sh / sum0;
        }
        row.push(clamp6(q));
      }
      for (let j = 1; j < 5; j++) if (row[j] < row[j - 1]) row[j] = row[j - 1];
      rows.push(row);
    }
    forecasts.push({ id: c.markets[mi].id, q: rows });
  }
  return { forecasts };
}

/** Entrypoint: read every .in in the input directory, write one .out per case. */
export function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
  mkdirSync(outputDir, { recursive: true });
  if (!existsSync(inputDir)) return;
  const files = readdirSync(inputDir).filter((f) => f.endsWith('.in')).sort();
  for (const file of files) {
    const id = file.slice(0, -'.in'.length);
    const text = readFileSync(join(inputDir, file), 'utf8');
    const c = JSON.parse(text) as CaseFile;
    if (!c || !Array.isArray(c.markets) || !Number.isInteger(c.horizon_minutes)) {
      throw new Error(`${file}: not a case`);
    }
    const out = forecastCase(c);
    writeFileSync(join(outputDir, `${id}.out`), `${JSON.stringify(out)}\n`);
  }
}

if (!process.env.MAKEFUN_NO_RUN) main();
