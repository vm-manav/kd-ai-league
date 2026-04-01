import { createRemoteJWKSet, JWTPayload, jwtVerify } from "jose";
import { config } from "../config.js";
import { AuthContext, Scope } from "../types.js";
import { resolveTierFromClaims } from "./tiers.js";

const jwks =
  config.oauth.jwksUri && config.oauth.jwksUri.length > 0
    ? createRemoteJWKSet(new URL(config.oauth.jwksUri))
    : null;

function parseScopes(payload: JWTPayload): Set<Scope> {
  const scopeString =
    typeof payload.scope === "string"
      ? payload.scope
      : Array.isArray(payload.scp)
        ? payload.scp.join(" ")
        : "";

  const raw = scopeString.split(/\s+/).filter(Boolean);
  const supported = new Set<Scope>([
    "market:read",
    "fundamentals:read",
    "technicals:read",
    "mf:read",
    "news:read",
    "filings:read",
    "filings:deep",
    "macro:read",
    "macro:historical",
    "research:generate",
    "watchlist:read",
    "watchlist:write",
    "portfolio:read",
    "portfolio:write",
  ]);

  const result = new Set<Scope>();
  for (const item of raw) {
    if (supported.has(item as Scope)) {
      result.add(item as Scope);
    }
  }
  return result;
}

export async function validateBearerToken(
  authorizationHeader?: string,
): Promise<AuthContext | null> {
  if (!authorizationHeader?.startsWith("Bearer ")) {
    return null;
  }

  const token = authorizationHeader.slice("Bearer ".length).trim();
  if (!token) {
    return null;
  }

  if (config.authBypass) {
    return {
      userId: "dev-user",
      tier: "analyst",
      scopes: new Set<Scope>([
        "market:read",
        "fundamentals:read",
        "technicals:read",
        "mf:read",
        "news:read",
        "filings:read",
        "filings:deep",
        "macro:read",
        "macro:historical",
        "research:generate",
        "watchlist:read",
        "watchlist:write",
      ]),
      tokenAudience: config.oauth.audience,
    };
  }

  if (!jwks || !config.oauth.issuer) {
    throw new Error("OAuth verifier misconfigured. Set OAUTH_ISSUER and OAUTH_JWKS_URI.");
  }

  const { payload } = await jwtVerify(token, jwks, {
    issuer: config.oauth.issuer,
    audience: config.oauth.audience,
  });

  const userId =
    (typeof payload.preferred_username === "string" && payload.preferred_username) ||
    (typeof payload.sub === "string" && payload.sub) ||
    "unknown-user";

  return {
    userId,
    tier: resolveTierFromClaims(payload as Record<string, unknown>),
    scopes: parseScopes(payload),
    tokenAudience: payload.aud,
  };
}
