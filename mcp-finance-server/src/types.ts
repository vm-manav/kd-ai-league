export type Tier = "free" | "premium" | "analyst";

export type Scope =
  | "market:read"
  | "fundamentals:read"
  | "technicals:read"
  | "mf:read"
  | "news:read"
  | "filings:read"
  | "filings:deep"
  | "macro:read"
  | "macro:historical"
  | "research:generate"
  | "watchlist:read"
  | "watchlist:write"
  | "portfolio:read"
  | "portfolio:write";

export interface AuthContext {
  userId: string;
  tier: Tier;
  scopes: Set<Scope>;
  tokenAudience: string | string[] | undefined;
}

export interface SourceCitation {
  source: string;
  detail: string;
  timestamp?: string;
}
