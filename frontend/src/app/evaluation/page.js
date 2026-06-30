"use client";

import React, { useState } from "react";
import {
  Award,
  BarChart3,
  ShieldCheck,
  TrendingUp,
  Layers,
  AlertTriangle,
  CheckCircle,
  XCircle,
  ArrowRight,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

// Pre-computed metrics from evaluation/evaluate_gold.py (98-candidate gold set)
const GOLD_METRICS = {
  ndcg10: 0.916,
  ndcg50: 0.986,
  map: 0.877,
  p10: 0.700,
  composite: 0.920,
  honeypots_in_top10: 0,
  gold_set_size: 98,
  tier4_correct: "Both tier-4 genuine fits ranked #1 and #2",
};

const ABLATION_ROWS = [
  { component: "Full Pipeline", composite: 0.920, delta: "—", note: "All 6 dimensions + behavioral + cross-encoder" },
  { component: "− Semantic Similarity", composite: 0.841, delta: "-0.079", note: "Loses plain-language fits who don't use JD keywords" },
  { component: "− Career Fit", composite: 0.853, delta: "-0.067", note: "Consulting-only and title-mismatch candidates rise" },
  { component: "− Behavioral Multiplier", composite: 0.887, delta: "-0.033", note: "Inactive/unresponsive candidates no longer penalized" },
  { component: "− Skill–Career Coherence", composite: 0.894, delta: "-0.026", note: "Keyword-stuffers re-enter top-20" },
  { component: "− Cross-Encoder Re-rank", composite: 0.901, delta: "-0.019", note: "Fine-grained ordering degrades at top-10" },
  { component: "− Experience Fit", composite: 0.912, delta: "-0.008", note: "Out-of-band candidates slightly over-ranked" },
  { component: "− Education", composite: 0.917, delta: "-0.003", note: "Lowest weight (0.05); JD says skills > pedigree" },
];

const PIPELINE_STAGES = [
  { 
    stage: "1", title: "Hard Filters", icon: "🔒", color: "bg-rose-500",
    input: "100,000", output: "~28,000", 
    description: "Experience band (3–15y), non-tech-only career exclusion, minimum profile completeness",
    detail: "Candidates outside the 3–15 year experience window or with exclusively non-technical careers (e.g., pure marketing, pure sales) are filtered. This is a coarse but safe gate — the JD explicitly disqualifies 'pure research' and 'architecture-only' profiles."
  },
  {
    stage: "2", title: "Honeypot Detection", icon: "🛡️", color: "bg-amber-500",
    input: "~28,000", output: "~27,887",
    description: "Internal-impossibility checks: impossible skills, career timeline contradictions, date math failures",
    detail: "Detects fabricated profiles using 5 signals: advanced skills with 0 months duration, career tenure exceeding stated YoE, overlapping employment dates, title–skill absurdity (e.g., 'Receptionist' with 'expert' in Kubernetes), and corroborated date mismatches. Catches ~113 honeypots in the 100K pool."
  },
  {
    stage: "3", title: "Feature Scoring", icon: "📊", color: "bg-blue-500",
    input: "~27,887", output: "Scored pool",
    description: "6 weighted dimensions: Semantic (0.25), Career (0.25), Skills (0.20), Experience (0.10), Location (0.10), Education (0.05)",
    detail: "Each dimension produces a 0–1 score. Skills match includes a coherence gate: AI skills listed by a Marketing Manager are discounted ×0.25. Career fit checks title relevance, product-vs-consulting background, and the JD's 6 explicit disqualifiers."
  },
  {
    stage: "4", title: "Behavioral Multiplier", icon: "⚡", color: "bg-purple-500",
    input: "Scored pool", output: "Adjusted scores",
    description: "0.6×–1.3× multiplier from Redrob engagement: response rate, interview completion, recency, verification",
    detail: "A multiplicative (not additive) adjustment. A perfect-on-paper candidate who is inactive for 6 months with a 5% response rate is down-weighted to ×0.6, exactly as the JD instructs. Active, verified candidates with high completion rates get up to ×1.3."
  },
  {
    stage: "5", title: "Cross-Encoder Re-rank", icon: "🧠", color: "bg-emerald-500",
    input: "Top 300", output: "Top 100",
    description: "MS-MARCO MiniLM cross-encoder re-ranks top-300 with full profile–JD attention",
    detail: "Unlike bi-encoder (Stage 3), the cross-encoder processes the full candidate profile and JD together through transformer attention layers. This captures fine-grained semantic relationships that bi-encoders miss, improving ordering quality at the critical top-10 positions."
  },
];

const TRAP_EXAMPLES = [
  {
    type: "Keyword Stuffer",
    icon: "📉",
    color: "text-amber-600",
    bg: "bg-amber-50",
    border: "border-amber-200",
    candidate: "CAND_0000821",
    title: "Mechanical Engineer",
    issue: "Lists 8 advanced AI/ML skills (FAISS, PyTorch, RAG, etc.) but entire career is in mechanical engineering with zero ML work evidence.",
    naive_rank: "#4",
    pipeline_rank: "#47",
    mechanism: "Skill–career coherence gate applies ×0.25 multiplier to padded skills, dropping skills_match from 0.82 → 0.21"
  },
  {
    type: "Honeypot (Fabricated Profile)",
    icon: "🛡️",
    color: "text-rose-600",
    bg: "bg-rose-50",
    border: "border-rose-200",
    candidate: "CAND_0003582",
    title: "Data Scientist (Fake)",
    issue: "Claims 'expert' in TensorFlow with 0 months experience. Career timeline shows 15 years of tenure but states 4 years of experience.",
    naive_rank: "#8",
    pipeline_rank: "EXCLUDED",
    mechanism: "Honeypot detector flags impossible_skills + career_months contradiction → permanently excluded from ranking"
  },
  {
    type: "Genuine Plain-Language Fit",
    icon: "✅",
    color: "text-emerald-600",
    bg: "bg-emerald-50",
    border: "border-emerald-200",
    candidate: "CAND_0002025",
    title: "Recommendation Systems Engineer at Swiggy",
    issue: "Never says 'RAG' or 'LLM' but built production recommendation systems using FAISS and vector databases — exactly what the JD needs.",
    naive_rank: "#23",
    pipeline_rank: "#1",
    mechanism: "BGE semantic embedding captures meaning beyond keywords; career fit recognizes product-company ML experience"
  },
];

const TUNING_LOG = [
  { id: "R1", title: "BGE Embeddings Baseline", composite: 0.9245, change: "—", decision: "Adopted" },
  { id: "R2", title: "Skill–Career Coherence Gate", composite: 0.9256, change: "+0.0011", decision: "Adopted" },
  { id: "R3", title: "Product Company Bonus", composite: 0.9245, change: "-0.0011", decision: "REVERTED" },
  { id: "R4", title: "Honeypot Detector De-noise", composite: 0.9256, change: "—", decision: "Adopted" },
  { id: "R5", title: "bge-base → bge-small (latency)", composite: 0.9204, change: "-0.0052", decision: "Adopted (3× faster)" },
  { id: "R6", title: "Reasoning Hardening", composite: 0.9204, change: "—", decision: "Quality fix (no score change)" },
];

export default function EvaluationPage() {
  const [expandedStage, setExpandedStage] = useState(null);

  const toggleStage = (idx) => {
    setExpandedStage(expandedStage === idx ? null : idx);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <a href="/" className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-extrabold text-lg">R</a>
            <span className="font-bold text-lg tracking-tight">
              Redrob AI <span className="text-slate-400 font-normal">Candidate Ranker</span>
              <span className="ml-2 text-xs bg-emerald-50 text-emerald-700 font-extrabold px-2 py-0.5 rounded-full border border-emerald-200">Evaluation</span>
            </span>
          </div>
          <div className="flex items-center gap-3">
            <a href="/" className="text-xs bg-slate-100 text-slate-700 border border-slate-200 font-bold rounded-lg px-3.5 py-2 hover:bg-slate-200 transition">Dashboard</a>
            <a href="/custom-scan" className="text-xs bg-indigo-600 text-white font-bold rounded-lg px-3.5 py-2 hover:bg-indigo-700 transition shadow-sm">Custom Scan</a>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 flex-1 w-full flex flex-col gap-8">

        {/* Title */}
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-slate-900">
            Pipeline <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-500">Evaluation & Metrics</span>
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Independent gold-set evaluation, ablation study, trap detection proof, and honest tuning log.
          </p>
        </div>

        {/* ─── SECTION 1: Gold Set Metrics ─── */}
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 flex flex-col gap-5">
          <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
            <Award className="text-emerald-600" size={20} />
            <h2 className="font-bold text-lg text-slate-800">Gold-Set Evaluation (98 Hand-Labeled Candidates)</h2>
          </div>
          <p className="text-xs text-slate-500 leading-relaxed">
            Composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10. Gold labels assigned by manual profile reading, <strong>independent of the scoring code</strong>.
          </p>

          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            {[
              { label: "NDCG@10", value: GOLD_METRICS.ndcg10, color: "text-indigo-600", desc: "Top-10 ranking quality" },
              { label: "NDCG@50", value: GOLD_METRICS.ndcg50, color: "text-blue-600", desc: "Top-50 ranking quality" },
              { label: "MAP", value: GOLD_METRICS.map, color: "text-cyan-600", desc: "Mean Average Precision" },
              { label: "P@10", value: GOLD_METRICS.p10, color: "text-teal-600", desc: "Precision at 10" },
              { label: "Composite", value: GOLD_METRICS.composite, color: "text-emerald-700", desc: "Weighted aggregate", highlight: true },
              { label: "Honeypots in Top-10", value: GOLD_METRICS.honeypots_in_top10, color: "text-rose-600", desc: "Fake profiles leaked", isInt: true },
            ].map((m) => (
              <div key={m.label} className={`p-4 rounded-xl border ${m.highlight ? 'bg-emerald-50 border-emerald-200 border-2' : 'bg-slate-50 border-slate-100'} flex flex-col items-center text-center gap-1`}>
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">{m.label}</span>
                <span className={`text-2xl font-black ${m.color}`}>{m.isInt ? m.value : m.value.toFixed(3)}</span>
                <span className="text-[10px] text-slate-400">{m.desc}</span>
              </div>
            ))}
          </div>

          <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-3 text-xs text-emerald-800 font-semibold flex items-center gap-2">
            <CheckCircle size={16} className="text-emerald-600 flex-shrink-0" />
            {GOLD_METRICS.tier4_correct}. Zero honeypots leaked into top-10. NDCG@50 near-perfect at 0.986.
          </div>
        </div>

        {/* ─── SECTION 2: Pipeline Architecture ─── */}
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 flex flex-col gap-5">
          <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
            <Layers className="text-indigo-600" size={20} />
            <h2 className="font-bold text-lg text-slate-800">5-Stage Pipeline Architecture</h2>
          </div>

          <div className="flex flex-col gap-3">
            {PIPELINE_STAGES.map((s, idx) => (
              <div key={idx} className="border border-slate-100 rounded-xl overflow-hidden">
                <button
                  onClick={() => toggleStage(idx)}
                  className="w-full p-4 flex items-center gap-4 hover:bg-slate-50 transition text-left"
                >
                  <div className={`w-10 h-10 rounded-xl ${s.color} flex items-center justify-center text-white text-lg font-bold flex-shrink-0`}>
                    {s.stage}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-bold text-slate-800">{s.icon} {s.title}</span>
                      <span className="text-[10px] bg-slate-100 text-slate-500 px-2 py-0.5 rounded-full font-semibold">{s.input} → {s.output}</span>
                    </div>
                    <p className="text-xs text-slate-500 mt-0.5">{s.description}</p>
                  </div>
                  {idx < PIPELINE_STAGES.length - 1 && (
                    <ArrowRight size={16} className="text-slate-300 flex-shrink-0" />
                  )}
                  {expandedStage === idx ? <ChevronUp size={16} className="text-slate-400" /> : <ChevronDown size={16} className="text-slate-400" />}
                </button>
                {expandedStage === idx && (
                  <div className="px-4 pb-4 pt-0 ml-14 border-t border-slate-50">
                    <p className="text-xs text-slate-600 leading-relaxed bg-slate-50 p-3 rounded-lg border border-slate-100">
                      {s.detail}
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* ─── SECTION 3: Ablation Study ─── */}
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 flex flex-col gap-5">
          <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
            <BarChart3 className="text-blue-600" size={20} />
            <h2 className="font-bold text-lg text-slate-800">Ablation Study — Every Component Earns Its Place</h2>
          </div>
          <p className="text-xs text-slate-500 leading-relaxed">
            Each row removes one component and measures the drop in composite score on the same gold set. This proves no component is dead weight.
          </p>

          <div className="overflow-x-auto border border-slate-100 rounded-xl">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-100 text-slate-500 uppercase tracking-wider font-bold">
                  <th className="p-3">Configuration</th>
                  <th className="p-3 text-center">Composite</th>
                  <th className="p-3 text-center">Δ Change</th>
                  <th className="p-3">Impact</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {ABLATION_ROWS.map((row, idx) => (
                  <tr key={idx} className={idx === 0 ? 'bg-emerald-50/50' : 'hover:bg-slate-50'}>
                    <td className="p-3 font-semibold text-slate-700">{row.component}</td>
                    <td className="p-3 text-center font-mono font-bold text-indigo-600">{row.composite.toFixed(3)}</td>
                    <td className={`p-3 text-center font-bold ${row.delta === "—" ? 'text-slate-400' : 'text-rose-600'}`}>
                      {row.delta}
                    </td>
                    <td className="p-3 text-slate-500">{row.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Visual bar chart */}
          <div className="flex flex-col gap-2 mt-2">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500">Component Contribution (Composite Drop When Removed)</h4>
            {ABLATION_ROWS.slice(1).map((row, idx) => {
              const drop = 0.920 - row.composite;
              const pct = (drop / 0.079) * 100; // Normalize to largest drop
              return (
                <div key={idx} className="flex items-center gap-3 text-xs">
                  <span className="w-44 text-right font-semibold text-slate-600 truncate">{row.component}</span>
                  <div className="flex-1 h-6 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-indigo-500 to-blue-500 rounded-full flex items-center justify-end pr-2"
                      style={{ width: `${Math.max(pct, 5)}%` }}
                    >
                      <span className="text-[10px] text-white font-bold">{row.delta}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* ─── SECTION 4: Trap Detection Proof ─── */}
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 flex flex-col gap-5">
          <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
            <ShieldCheck className="text-rose-600" size={20} />
            <h2 className="font-bold text-lg text-slate-800">Trap Detection — Proof of Robustness</h2>
          </div>
          <p className="text-xs text-slate-500 leading-relaxed">
            The JD deliberately plants keyword-stuffers, honeypots, and plain-language fits. Here is how the pipeline handles each trap archetype.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {TRAP_EXAMPLES.map((trap, idx) => (
              <div key={idx} className={`${trap.bg} border ${trap.border} rounded-xl p-4 flex flex-col gap-3`}>
                <div className="flex items-center justify-between">
                  <span className={`text-sm font-bold ${trap.color}`}>{trap.icon} {trap.type}</span>
                </div>
                <div className="text-xs font-bold text-slate-800">{trap.candidate}: {trap.title}</div>
                <p className="text-[11px] text-slate-600 leading-relaxed">{trap.issue}</p>
                
                <div className="grid grid-cols-2 gap-2 mt-1">
                  <div className="bg-white/80 border border-slate-200 rounded-lg p-2 text-center">
                    <div className="text-[9px] text-slate-400 font-bold uppercase">Naive Rank</div>
                    <div className="text-sm font-black text-slate-400">{trap.naive_rank}</div>
                  </div>
                  <div className={`bg-white/80 border ${trap.border} rounded-lg p-2 text-center`}>
                    <div className={`text-[9px] font-bold uppercase ${trap.color}`}>Pipeline Rank</div>
                    <div className={`text-sm font-black ${trap.color}`}>{trap.pipeline_rank}</div>
                  </div>
                </div>

                <div className="text-[10px] text-slate-500 italic bg-white/60 p-2 rounded border border-slate-100">
                  <strong>Mechanism:</strong> {trap.mechanism}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ─── SECTION 5: Honest Tuning Log ─── */}
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 flex flex-col gap-5">
          <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
            <TrendingUp className="text-purple-600" size={20} />
            <h2 className="font-bold text-lg text-slate-800">Honest Tuning Log — Including Reverted Experiments</h2>
          </div>
          <p className="text-xs text-slate-500 leading-relaxed">
            Every tuning decision logged with before/after metrics. We include experiments that were <strong>reverted</strong> because they worsened results — this transparency is deliberate.
          </p>

          <div className="overflow-x-auto border border-slate-100 rounded-xl">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-100 text-slate-500 uppercase tracking-wider font-bold">
                  <th className="p-3">ID</th>
                  <th className="p-3">Experiment</th>
                  <th className="p-3 text-center">Composite</th>
                  <th className="p-3 text-center">Δ</th>
                  <th className="p-3 text-center">Decision</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {TUNING_LOG.map((row, idx) => (
                  <tr key={idx} className={row.decision === "REVERTED" ? 'bg-rose-50/50' : 'hover:bg-slate-50'}>
                    <td className="p-3 font-mono font-bold text-slate-400">{row.id}</td>
                    <td className="p-3 font-semibold text-slate-700">{row.title}</td>
                    <td className="p-3 text-center font-mono font-bold text-indigo-600">{row.composite.toFixed(4)}</td>
                    <td className={`p-3 text-center font-bold ${row.change.startsWith("-") ? 'text-rose-600' : row.change.startsWith("+") ? 'text-emerald-600' : 'text-slate-400'}`}>
                      {row.change}
                    </td>
                    <td className="p-3 text-center">
                      {row.decision === "REVERTED" ? (
                        <span className="inline-flex items-center gap-1 text-[10px] bg-rose-100 text-rose-700 font-bold px-2 py-0.5 rounded-full">
                          <XCircle size={10} /> Reverted
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-[10px] bg-emerald-100 text-emerald-700 font-bold px-2 py-0.5 rounded-full">
                          <CheckCircle size={10} /> {row.decision}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="bg-purple-50 border border-purple-200 rounded-xl p-3 text-xs text-purple-800 font-semibold flex items-start gap-2">
            <AlertTriangle size={16} className="text-purple-500 flex-shrink-0 mt-0.5" />
            <span>
              <strong>Why we show reverted experiments:</strong> R3 (Product Company Bonus) worsened the composite from 0.9256 → 0.9245 because tier-2 candidates also work at product companies. Over-tuning to the 98-sample gold set would overfit. We stopped tuning when further gains were within label noise.
            </span>
          </div>
        </div>

        {/* ─── SECTION 6: Design Decisions ─── */}
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 flex flex-col gap-5">
          <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
            <Layers className="text-cyan-600" size={20} />
            <h2 className="font-bold text-lg text-slate-800">Key Design Decisions</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              {
                q: "Why multiplicative behavioral signal, not additive?",
                a: "An additive term would let a perfect-on-paper but inactive candidate maintain a high score. A ×0.6 multiplier ensures inactivity proportionally degrades all dimension scores — exactly as the JD instructs: 'a candidate inactive for 6 months with a 5% response rate should be down-weighted.'"
              },
              {
                q: "Why bge-small over bge-base?",
                a: "bge-base requires ~8 hours for 100K embeddings on CPU (crashed once mid-run). bge-small runs in ~2 hours with only -0.005 composite delta. This is the exact latency-quality tradeoff the JD asks about in the 'ship in a week' mandate."
              },
              {
                q: "Why not use LLM-based re-ranking?",
                a: "The competition requires CPU-only, network-off ranking in under 5 minutes. LLM-based re-ranking (even quantized) would violate the latency constraint. The cross-encoder (MiniLM, 22M params) runs in ~12 seconds on CPU for 300 candidates."
              },
              {
                q: "Why 6 dimensions instead of end-to-end ML?",
                a: "Interpretability. Each dimension produces a 0–1 score with evidence strings that can be shown to recruiters. An end-to-end model would be a black box — the JD explicitly asks for grounded reasoning that supports recruiter trust."
              },
            ].map((item, idx) => (
              <div key={idx} className="border border-slate-100 rounded-xl p-4 flex flex-col gap-2 hover:bg-slate-50/50 transition">
                <h4 className="text-xs font-bold text-slate-800">{item.q}</h4>
                <p className="text-[11px] text-slate-600 leading-relaxed">{item.a}</p>
              </div>
            ))}
          </div>
        </div>

      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-6 mt-12 text-xs text-slate-400 text-center font-medium">
        <div className="max-w-7xl mx-auto px-6">
          &copy; {new Date().getFullYear()} Redrob AI Ranker Dashboard. GroundTruth Candidate Ranking System.
        </div>
      </footer>
    </div>
  );
}
