import { Tier } from "../types.js";

export interface AuditEntry {
  userId: string;
  tier: Tier;
  operation: string;
  status: "success" | "error" | "forbidden";
  timestamp: string;
  details?: Record<string, unknown>;
}

const entries: AuditEntry[] = [];

export function appendAudit(entry: AuditEntry): void {
  entries.push(entry);
  if (entries.length > 5000) {
    entries.shift();
  }
  // Hackathon-friendly visibility; replace with durable sink in production.
  console.log("[AUDIT]", JSON.stringify(entry));
}

export function recentAuditEntries(limit = 100): AuditEntry[] {
  return entries.slice(-Math.max(1, limit));
}
