import { Tier } from "../types.js";
import { config } from "../config.js";

const rank: Record<Tier, number> = {
  free: 1,
  premium: 2,
  analyst: 3,
};

export function hasMinimumTier(actual: Tier, required: Tier): boolean {
  return rank[actual] >= rank[required];
}

export function resolveTierFromClaims(payload: Record<string, unknown>): Tier {
  const namespacedTier = payload[config.oauth.tierClaim];
  if (
    namespacedTier === "free" ||
    namespacedTier === "premium" ||
    namespacedTier === "analyst"
  ) {
    return namespacedTier;
  }

  const explicitTier = payload.tier;
  if (explicitTier === "free" || explicitTier === "premium" || explicitTier === "analyst") {
    return explicitTier;
  }

  const realmAccess = payload.realm_access as { roles?: string[] } | undefined;
  const namespacedRoles = Array.isArray(payload[config.oauth.rolesClaim])
    ? (payload[config.oauth.rolesClaim] as string[])
    : [];
  const roles = [...namespacedRoles, ...(realmAccess?.roles ?? [])];

  if (roles.includes("analyst") || roles.includes("finance-analyst")) {
    return "analyst";
  }
  if (roles.includes("premium") || roles.includes("finance-premium")) {
    return "premium";
  }
  return "free";
}
