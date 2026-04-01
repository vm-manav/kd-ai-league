import { Response } from "express";
import { config } from "../config.js";

function metadataUrl(): string {
  return `${config.serverBaseUrl}/.well-known/oauth-protected-resource`;
}

export function sendUnauthorized(res: Response): void {
  res.setHeader(
    "WWW-Authenticate",
    `Bearer realm="mcp-finance-server", resource_metadata="${metadataUrl()}"`,
  );
  res.status(401).json({
    error: "unauthorized",
    message: "Authentication required.",
    resource_metadata: metadataUrl(),
  });
}

export function sendForbidden(res: Response, requiredScopes: string[]): void {
  res.setHeader(
    "WWW-Authenticate",
    `Bearer error="insufficient_scope", scope="${requiredScopes.join(" ")}"`,
  );
  res.status(403).json({
    error: "insufficient_scope",
    required_scopes: requiredScopes,
  });
}

export function protectedResourceMetadata(): Record<string, unknown> {
  return {
    resource: config.serverBaseUrl,
    authorization_servers: [config.oauth.authorizationServer],
    bearer_methods_supported: ["header"],
    scopes_supported: [
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
    ],
  };
}
