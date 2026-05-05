import type {
  AgentResult,
  Category,
  ScrapedSite,
} from "@/types";
import { callLlm, type ProviderChoice } from "@/lib/llm-router";

// ============================================================
// Shared helpers (kept inline per-file to minimize cross-file deps)
// ============================================================

export function summarizeSiteForPrompt(site: ScrapedSite): string {
  const h = site.homepage;
  const headings = h.headings
    .slice(0, 20)
    .map((x) => `H${x.level}: ${x.text}`)
    .join("\n");
  const ctas = h.cta_buttons.slice(0, 25).join(" | ");
  const subPages = site.pages
    .map((p) => `- ${p.url} — "${p.title}" — ${p.meta_description.slice(0, 120)}`)
    .join("\n");

  return [
    `URL: ${h.url}`,
    `Title: ${h.title}`,
    `Meta description: ${h.meta_description}`,
    `Business type (heuristic): ${site.business_type}`,
    `Languages: ${site.detected_languages.join(", ")}`,
    `Has pricing page: ${site.has_pricing_page}`,
    `Has about page: ${site.has_about_page}`,
    `Has blog: ${site.has_blog}`,
    `Has contact form: ${site.has_contact_form}`,
    `Has chat widget: ${h.has_chat_widget}`,
    "",
    "Headings (homepage):",
    headings || "(none)",
    "",
    "CTAs (homepage):",
    ctas || "(none)",
    "",
    "Sub-pages discovered:",
    subPages || "(none)",
    "",
    "Homepage visible text (truncated to 3000 chars):",
    h.visible_text.slice(0, 3000),
  ].join("\n");
}

export function buildJsonSchemaInstruction(category: Category): string {
  return [
    "LANGUE DE SORTIE: tous les champs textuels (strengths, weaknesses, findings.title, findings.description, recommendations.title, recommendations.description, recommendations.timeline, summary) DOIVENT être rédigés en FRANÇAIS, peu importe la langue du site analysé.",
    "",
    "Tu DOIS retourner UNIQUEMENT un objet JSON valide — aucun texte avant ou après, aucun bloc markdown ```json.",
    "Schéma strict :",
    "{",
    `  "category": "${category}",`,
    '  "score": <entier 0-100>,',
    '  "strengths": [<string en français>],',
    '  "weaknesses": [<string en français>],',
    '  "findings": [ { "title": <string FR>, "severity": "critical"|"high"|"medium"|"low", "description": <string FR>, "evidence": <citation verbatim du site, peut rester dans la langue d\'origine> } ],',
    '  "recommendations": [ { "priority": "quick_win"|"strategic"|"long_term", "title": <string FR>, "description": <string FR>, "impact_estimate_min": <number EUR/mois>, "impact_estimate_max": <number EUR/mois>, "timeline": <string FR ex "1 semaine">, "confidence": "high"|"medium"|"low" } ],',
    '  "summary": <2-3 phrases en français>',
    "}",
  ].join("\n");
}

export function tryParseAgentResult(
  raw: string,
  category: Category
): AgentResult {
  // Extract first {...} block defensively (some models wrap in fences).
  const cleaned = raw
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();
  let jsonText = cleaned;
  const first = cleaned.indexOf("{");
  const last = cleaned.lastIndexOf("}");
  if (first >= 0 && last > first) jsonText = cleaned.slice(first, last + 1);

  let obj: unknown;
  try {
    obj = JSON.parse(jsonText);
  } catch (err) {
    throw new Error(
      `[agent:${category}] Could not parse JSON: ${(err as Error).message}. Raw head: ${raw.slice(0, 200)}`
    );
  }
  return normalizeAgentResult(obj, category);
}

function clampScore(n: unknown): number {
  const v = typeof n === "number" ? n : Number(n);
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(100, Math.round(v)));
}

function asStringArray(x: unknown): string[] {
  if (!Array.isArray(x)) return [];
  return x.filter((v): v is string => typeof v === "string" && v.length > 0);
}

function normalizeAgentResult(obj: unknown, category: Category): AgentResult {
  const o = (obj ?? {}) as Record<string, unknown>;
  const findingsRaw = Array.isArray(o.findings) ? o.findings : [];
  const recsRaw = Array.isArray(o.recommendations) ? o.recommendations : [];

  return {
    category,
    score: clampScore(o.score),
    strengths: asStringArray(o.strengths),
    weaknesses: asStringArray(o.weaknesses),
    findings: findingsRaw.map((f) => {
      const fo = (f ?? {}) as Record<string, unknown>;
      const sev = String(fo.severity ?? "medium").toLowerCase();
      const severity =
        sev === "critical" || sev === "high" || sev === "medium" || sev === "low"
          ? (sev as "critical" | "high" | "medium" | "low")
          : "medium";
      return {
        title: String(fo.title ?? ""),
        severity,
        description: String(fo.description ?? ""),
        evidence: String(fo.evidence ?? ""),
      };
    }),
    recommendations: recsRaw.map((r) => {
      const ro = (r ?? {}) as Record<string, unknown>;
      const prio = String(ro.priority ?? "strategic").toLowerCase();
      const priority =
        prio === "quick_win" || prio === "strategic" || prio === "long_term"
          ? (prio as "quick_win" | "strategic" | "long_term")
          : "strategic";
      const conf = String(ro.confidence ?? "medium").toLowerCase();
      const confidence =
        conf === "high" || conf === "medium" || conf === "low"
          ? (conf as "high" | "medium" | "low")
          : "medium";
      return {
        priority,
        title: String(ro.title ?? ""),
        description: String(ro.description ?? ""),
        impact_estimate_min: Number(ro.impact_estimate_min ?? 0) || 0,
        impact_estimate_max: Number(ro.impact_estimate_max ?? 0) || 0,
        timeline: String(ro.timeline ?? ""),
        confidence,
      };
    }),
    summary: String(o.summary ?? ""),
  };
}

/**
 * Run a single category agent against multiple providers. When more than one
 * provider is given (premium tier), scores are averaged and findings/recos
 * merged & deduped by title.
 */
export async function runAgentWithProviders(
  category: Category,
  systemPrompt: string,
  userPrompt: string,
  providers: ProviderChoice[]
): Promise<AgentResult> {
  if (providers.length === 0) {
    throw new Error(`[agent:${category}] No providers supplied.`);
  }

  const results: AgentResult[] = [];
  for (const p of providers) {
    const result = await callWithRetry(
      p,
      category,
      systemPrompt,
      userPrompt
    );
    results.push(result);
  }

  if (results.length === 1) return results[0];

  // Consensus: average score, merge unique findings/recos by title.
  const avgScore = clampScore(
    results.reduce((s, r) => s + r.score, 0) / results.length
  );
  const seenF = new Set<string>();
  const seenR = new Set<string>();
  const merged: AgentResult = {
    category,
    score: avgScore,
    strengths: dedupe(results.flatMap((r) => r.strengths)),
    weaknesses: dedupe(results.flatMap((r) => r.weaknesses)),
    findings: results
      .flatMap((r) => r.findings)
      .filter((f) => {
        const k = f.title.trim().toLowerCase();
        if (!k || seenF.has(k)) return false;
        seenF.add(k);
        return true;
      }),
    recommendations: results
      .flatMap((r) => r.recommendations)
      .filter((r) => {
        const k = r.title.trim().toLowerCase();
        if (!k || seenR.has(k)) return false;
        seenR.add(k);
        return true;
      }),
    summary: results.map((r) => r.summary).join(" "),
  };
  return merged;
}

/**
 * Call LLM with retry on parse failure. First attempt: full prompt @ temp 0.2.
 * Second attempt (on parse fail or empty result): same prompt + strict reminder
 * @ temp 0. Third attempt: minimal stub result if still failing (no throw — the
 * synthesizer treats missing categories as 0).
 */
async function callWithRetry(
  p: ProviderChoice,
  category: Category,
  systemPrompt: string,
  userPrompt: string
): Promise<AgentResult> {
  // Attempt 1
  try {
    const resp = await callLlm({
      provider: p.provider,
      model: p.model,
      system: systemPrompt,
      user: userPrompt,
      temperature: 0.2,
      max_tokens: 3000,
    });
    const parsed = tryParseAgentResult(resp.content, category);
    if (parsed.score > 0 || parsed.findings.length > 0 || parsed.recommendations.length > 0) {
      return parsed;
    }
    // Empty result — fall through to retry
    console.warn(`[agent:${category}] empty result on attempt 1, retrying...`);
  } catch (err) {
    console.warn(
      `[agent:${category}] attempt 1 failed (${(err as Error).message.slice(0, 120)}), retrying...`
    );
  }

  // Attempt 2: stricter prompt, temperature 0
  try {
    const stricterSystem =
      systemPrompt +
      "\n\nRAPPEL CRITIQUE : Ta réponse DOIT commencer par '{' et finir par '}'. Aucun texte avant ou après. Aucun ```json. Si tu ne respectes pas ce format, le système plante.";
    const resp = await callLlm({
      provider: p.provider,
      model: p.model,
      system: stricterSystem,
      user: userPrompt,
      temperature: 0,
      max_tokens: 3000,
    });
    return tryParseAgentResult(resp.content, category);
  } catch (err) {
    console.error(
      `[agent:${category}] BOTH attempts failed for ${p.provider}:${p.model} — falling back to stub. Last err: ${(err as Error).message.slice(0, 200)}`
    );
    // Stub result so the audit pipeline doesn't completely fail. The category
    // gets a 0 score and a single explanatory finding.
    return {
      category,
      score: 0,
      strengths: [],
      weaknesses: [`L'agent ${category} n'a pas pu produire de résultat exploitable (LLM ${p.provider}:${p.model}).`],
      findings: [],
      recommendations: [],
      summary: `Catégorie ${category} non évaluée — relancez l'audit ou passez en tier supérieur.`,
    };
  }
}

function dedupe(arr: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const s of arr) {
    const k = s.trim().toLowerCase();
    if (!k || seen.has(k)) continue;
    seen.add(k);
    out.push(s);
  }
  return out;
}

// ============================================================
// Content agent
// ============================================================

const CATEGORY: Category = "content";

const SYSTEM = `You are a senior marketing content auditor. Evaluate the website's CONTENT & MESSAGING quality.

Scoring rubric (0-100):
- Clarity of value proposition (is it instantly clear what the business does + for whom?)
- Headline strength on the homepage (specificity, benefit, hook)
- Tone consistency and brand voice
- Content depth (does it answer buyer questions? blog/articles?)
- Use of social proof IN COPY (testimonials, stats, named customers)
- Readability (jargon level, sentence length, scannability)

Penalize: vague taglines ("we help businesses grow"), missing H1, walls of text, generic stock copy, untranslated content, broken benefit/feature framing.

Be SPECIFIC and CITE evidence verbatim from the scraped content in the "evidence" field of each finding.

${buildJsonSchemaInstruction(CATEGORY)}`;

/**
 * Run the content & messaging agent against the scraped site.
 */
export async function runAgent(
  site: ScrapedSite,
  providers: ProviderChoice[]
): Promise<AgentResult> {
  return runAgentWithProviders(
    CATEGORY,
    SYSTEM,
    summarizeSiteForPrompt(site),
    providers
  );
}
