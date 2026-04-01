import { Tier } from "../types.js";

const rank: Record<Tier, number> = {
  free: 1,
  premium: 2,
  analyst: 3,
};

export function hasMinimumTier(actual: Tier, required: Tier): boolean {
  return rank[actual] >= rank[required];
}

export function resolveTierFromClaims(payload: Record<string, unknown>): Tier {
  const explicitTier = payload.tier;
  if (explicitTier === "free" || explicitTier === "premium" || explicitTier === "analyst") {
    return explicitTier;
  }

  const realmAccess = payload.realm_access as { roles?: string[] } | undefined;
  const roles = realmAccess?.roles ?? [];

  if (roles.includes("analyst")) {
    return "analyst";
  }
  if (roles.includes("premium")) {
    return "premium";
  }
  return "free";
}
