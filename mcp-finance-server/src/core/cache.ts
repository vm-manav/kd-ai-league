interface CacheEntry<T> {
  value: T;
  expiresAt: number;
}

const cache = new Map<string, CacheEntry<unknown>>();

export async function withTtlCache<T>(
  key: string,
  ttlSeconds: number,
  producer: () => Promise<T>,
): Promise<{ data: T; stale: boolean; cacheHit: boolean }> {
  const now = Date.now();
  const existing = cache.get(key) as CacheEntry<T> | undefined;

  if (existing && existing.expiresAt > now) {
    return { data: existing.value, stale: false, cacheHit: true };
  }

  try {
    const fresh = await producer();
    cache.set(key, { value: fresh, expiresAt: now + ttlSeconds * 1000 });
    return { data: fresh, stale: false, cacheHit: false };
  } catch (error) {
    if (existing) {
      return { data: existing.value, stale: true, cacheHit: true };
    }
    throw error;
  }
}
