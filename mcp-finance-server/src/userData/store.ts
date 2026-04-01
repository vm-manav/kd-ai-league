const watchlists = new Map<string, string[]>();
const researchBriefs = new Map<string, Record<string, unknown>>();

export function getWatchlist(userId: string): string[] {
  return watchlists.get(userId) ?? ["RELIANCE", "INFY", "HDFCBANK"];
}

export function saveWatchlist(userId: string, tickers: string[]): void {
  watchlists.set(userId, tickers.map((t) => t.toUpperCase()));
}

export function saveResearch(ticker: string, payload: Record<string, unknown>): void {
  researchBriefs.set(ticker.toUpperCase(), payload);
}

export function getResearch(ticker: string): Record<string, unknown> | null {
  return researchBriefs.get(ticker.toUpperCase()) ?? null;
}
