import { Tier } from "../types.js";

const tierLimits: Record<Tier, number> = {
  free: 30,
  premium: 150,
  analyst: 500,
};

interface Bucket {
  count: number;
  windowStartMs: number;
}

const buckets = new Map<string, Bucket>();
const WINDOW_MS = 60 * 60 * 1000;

export function checkAndConsumeRateLimit(input: {
  userId: string;
  tier: Tier;
}): { ok: boolean; retryAfterSec?: number; limit: number; remaining: number } {
  const now = Date.now();
  const limit = tierLimits[input.tier];
  const key = `${input.userId}:${input.tier}`;
  const current = buckets.get(key);

  if (!current || now - current.windowStartMs >= WINDOW_MS) {
    buckets.set(key, { count: 1, windowStartMs: now });
    return { ok: true, limit, remaining: limit - 1 };
  }

  if (current.count >= limit) {
    const retryAfterSec = Math.ceil((WINDOW_MS - (now - current.windowStartMs)) / 1000);
    return { ok: false, retryAfterSec, limit, remaining: 0 };
  }

  current.count += 1;
  buckets.set(key, current);
  return { ok: true, limit, remaining: limit - current.count };
}
