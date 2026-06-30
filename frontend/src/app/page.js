"use client";

import React, { useState, useEffect } from "react";
import { 
  Users, 
  ShieldCheck, 
  Flame, 
  Award, 
  Search, 
  FileText, 
  Download, 
  AlertTriangle, 
  ArrowUpRight, 
  ChevronRight,
  TrendingDown
} from "lucide-react";

export default function Home() {
  const [apiHost, setApiHost] = useState(
    typeof window !== "undefined"
      ? (process.env.NEXT_PUBLIC_API_HOST || "https://vishnups08-ai-candidate-ranking-system-backend.hf.space")
      : "https://vishnups08-ai-candidate-ranking-system-backend.hf.space"
  );
  
  const [status, setStatus] = useState(null);
  const [results, setResults] = useState([]);
  const [contrastData, setContrastData] = useState({ honeypot_cards: [], demotion_cards: [] });
  const [naiveCompare, setNaiveCompare] = useState({ naive_top20: [], pipeline_top20: [] });
  
  const [selectedRank, setSelectedRank] = useState(1);
  const [selectedCandidate, setSelectedCandidate] = useState(null);
  const [selectedTab, setSelectedTab] = useState("score");
  
  const [customJdText, setCustomJdText] = useState("");
  const [isReranking, setIsReranking] = useState(false);
  const [isApiLoading, setIsApiLoading] = useState(true);
  const [apiError, setApiError] = useState(null);

  // Fetch initial data
  const fetchData = async (host = apiHost) => {
    setIsApiLoading(true);
    setApiError(null);
    try {
      // 1. Fetch status
      const statusRes = await fetch(`${host}/api/status`);
      if (!statusRes.ok) throw new Error("Could not connect to backend");
      const statusData = await statusRes.json();
      setStatus(statusData);

      // 2. Fetch ranked results
      const rankRes = await fetch(`${host}/api/rank?n=50`);
      const rankData = await rankRes.json();
      setResults(rankData.results || []);

      // 3. Fetch contrast cases
      const contrastRes = await fetch(`${host}/api/contrast`);
      const contrastData = await contrastRes.json();
      setContrastData(contrastData);

      // 4. Fetch naive comparison
      const naiveRes = await fetch(`${host}/api/naive-compare`);
      const naiveData = await naiveRes.json();
      setNaiveCompare(naiveData);

      // Auto-select first candidate if results exist
      if (rankData.results && rankData.results.length > 0) {
        setSelectedRank(1);
        setSelectedCandidate(rankData.results[0]);
      }
    } catch (err) {
      console.error(err);
      setApiError("Hugging Face API server is not responding. Please wait for it to wake up or run: python demo_server.py --candidates precompute/sample_candidates.json --port 5050");
    } finally {
      setIsApiLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Update selected candidate details when rank selection changes
  useEffect(() => {
    if (results.length > 0) {
      const match = results.find(r => r.rank === selectedRank);
      if (match) {
        setSelectedCandidate(match);
      }
    }
  }, [selectedRank, results]);

  // Handle custom JD re-ranking
  const handleRerank = async () => {
    if (!customJdText.trim()) return;
    setIsReranking(true);
    try {
      const res = await fetch(`${apiHost}/api/rank-jd`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jd_text: customJdText, n: 50 }),
      });
      const data = await res.json();
      if (data.error) {
        alert(data.error);
        return;
      }
      
      // Update results dynamically
      setResults(data.results || []);
      
      // Refresh status details
      const statusRes = await fetch(`${apiHost}/api/status`);
      const statusData = await statusRes.json();
      setStatus(statusData);

      // Select top candidate
      if (data.results && data.results.length > 0) {
        setSelectedRank(1);
        setSelectedCandidate(data.results[0]);
      }
      
      alert("Pipeline successfully re-ranked candidates on-the-fly!");
    } catch (err) {
      console.error(err);
      alert("Failed to connect to Python backend for custom ranking.");
    } finally {
      setIsReranking(false);
    }
  };

  // Export CSV
  const handleDownloadCsv = () => {
    if (results.length === 0) return;
    let csvContent = "data:text/csv;charset=utf-8,Rank,Candidate ID,Score,Reasoning\n";
    results.forEach(r => {
      csvContent += `${r.rank},${r.candidate_id},${r.score},\"${r.reasoning.replace(/"/g, '""')}\"\n`;
    });
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "shortlist_export.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white font-extrabold text-lg">R</div>
            <span className="font-bold text-lg tracking-tight">Redrob AI <span className="text-slate-400 font-normal">Candidate Ranker</span></span>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Link to evaluation page */}
            <a 
              href="/evaluation"
              className="text-xs bg-emerald-600 text-white font-bold rounded-lg px-4 py-2 hover:bg-emerald-700 transition shadow-md shadow-emerald-100"
            >
              Evaluation &amp; Metrics
            </a>
            {/* Link to custom scan */}
            <a 
              href="/custom-scan"
              className="text-xs bg-indigo-600 text-white font-bold rounded-lg px-4 py-2 hover:bg-indigo-700 transition shadow-md shadow-indigo-100"
            >
              Custom File Scan
            </a>

            {/* Status indicator */}
            <div className="flex items-center gap-2 bg-slate-100 border border-slate-200 rounded-full px-3 py-1.5 text-xs font-semibold text-slate-700">
              <span className={`w-2.5 h-2.5 rounded-full ${apiError ? 'bg-rose-500 animate-pulse' : 'bg-emerald-500'}`} />
              <span>{apiError ? 'Cloud API Offline' : 'Cloud API Connected'}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8 flex-1 w-full flex flex-col gap-6">
        
        {/* Error Alert */}
        {apiError && (
          <div className="bg-rose-50 border border-rose-200 text-rose-800 rounded-xl p-4 flex gap-3 shadow-sm items-start">
            <AlertTriangle className="text-rose-500 flex-shrink-0 mt-0.5" size={20} />
            <div className="flex-1">
              <h4 className="font-bold text-sm">Python Backend Connection Error</h4>
              <p className="text-xs mt-1 text-rose-600 leading-relaxed font-semibold">{apiError}</p>
            </div>
            <button 
              onClick={() => fetchData()}
              className="text-xs bg-rose-600 text-white font-bold rounded-lg px-3 py-1.5 hover:bg-rose-700 transition"
            >
              Retry Connection
            </button>
          </div>
        )}

        {/* Dashboard Title & Description */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-extrabold tracking-tight text-slate-900">
              Candidate <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 via-blue-600 to-cyan-500">Discovery Engine</span>
            </h1>
            <p className="text-sm text-slate-500 mt-1">
              Full-context candidate discovery platform matching career histories, technical depth, and pedigree.
            </p>
          </div>
          <button 
            onClick={handleDownloadCsv}
            disabled={results.length === 0}
            className="flex items-center gap-2 bg-indigo-600 text-white px-4 py-2.5 rounded-xl font-semibold text-sm hover:bg-indigo-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition shadow-md shadow-indigo-200"
          >
            <Download size={16} />
            Export Shortlist CSV
          </button>
        </div>

        {/* Statistics Panels */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm border-l-4 border-l-indigo-600">
            <div className="flex items-center justify-between text-slate-400">
              <span className="text-xs font-bold uppercase tracking-wider text-indigo-600">Total Pool</span>
              <Users size={18} />
            </div>
            <div className="text-3xl font-black mt-2">{status?.total_candidates || 0}</div>
            <div className="text-xxs text-slate-500 mt-1 font-semibold">Loaded profiles</div>
          </div>
          
          <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm border-l-4 border-l-blue-600">
            <div className="flex items-center justify-between text-slate-400">
              <span className="text-xs font-bold uppercase tracking-wider text-blue-600">Passed Filters</span>
              <ShieldCheck size={18} />
            </div>
            <div className="text-3xl font-black mt-2">{status?.after_hard_filters || 0}</div>
            <div className="text-xxs text-emerald-600 mt-1 font-bold">
              Pass rate: {status?.total_candidates ? ((status.after_hard_filters / status.total_candidates) * 100).toFixed(1) : 0}%
            </div>
          </div>

          <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm border-l-4 border-l-rose-600">
            <div className="flex items-center justify-between text-slate-400">
              <span className="text-xs font-bold uppercase tracking-wider text-rose-600">Honeypots Blocked</span>
              <Flame size={18} />
            </div>
            <div className="text-3xl font-black mt-2">{status?.honeypots_detected || 0}</div>
            <div className="text-xxs text-rose-600 mt-1 font-bold">Excluded from score</div>
          </div>

          <div className="bg-white rounded-2xl p-5 border border-slate-200 shadow-sm border-l-4 border-l-emerald-600">
            <div className="flex items-center justify-between text-slate-400">
              <span className="text-xs font-bold uppercase tracking-wider text-emerald-600">Top Match Score</span>
              <Award size={18} />
            </div>
            <div className="text-3xl font-black mt-2">{(status?.top_score || 0).toFixed(3)}</div>
            <div className="text-xxs text-slate-500 mt-1 font-semibold">Max possible: 1.350</div>
          </div>
        </div>

        {/* Hero Feature: Custom JD Input */}
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
          <div className="flex items-center gap-2 mb-4">
            <Search className="text-indigo-600" size={18} />
            <h3 className="font-bold text-base text-slate-800">Verify Generalizability: Custom Job Description Re-ranking</h3>
          </div>
          <div className="flex flex-col gap-3">
            <textarea
              placeholder="Paste any custom Job Description here to test the pipeline on-the-fly (e.g. 'Backend Engineer with 5 years experience in Go and Milvus'). Candidate vectors will be scored against your custom input instantly."
              value={customJdText}
              onChange={(e) => setCustomJdText(e.target.value)}
              className="w-full h-24 border border-slate-200 rounded-xl p-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-600 placeholder:text-slate-400 bg-slate-50 focus:bg-white transition"
            />
            <div className="flex items-center justify-between gap-4">
              <span className="text-xs text-slate-500 leading-normal">
                🧠 Uses **BGE-small-en-v1.5** local cosine similarity + dynamic skill matching on-the-fly.
              </span>
              <button
                onClick={handleRerank}
                disabled={isReranking || !customJdText.trim() || apiError}
                className="bg-indigo-600 text-white font-bold text-sm px-5 py-2.5 rounded-xl hover:bg-indigo-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition shadow-md shadow-indigo-100 flex items-center gap-2"
              >
                {isReranking ? "Re-scoring Candidates..." : "Run Neural Re-rank"}
              </button>
            </div>
          </div>
        </div>

        {/* Main Grid: Shortlist list vs Selected candidate detail */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
          
          {/* Left: Shortlist & Table */}
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 flex flex-col gap-4">
            <h3 className="font-bold text-lg text-slate-800 border-b border-slate-100 pb-3 flex items-center justify-between">
              <span>🏆 Candidate Shortlist</span>
              <span className="text-xs font-normal text-slate-400">Total ranked: {results.length}</span>
            </h3>
            
            <div className="flex flex-col gap-3">
              <label className="text-xs text-slate-500 font-bold uppercase tracking-wider">Inspect Profile Details:</label>
              <select
                value={selectedRank}
                onChange={(e) => setSelectedRank(Number(e.target.value))}
                className="w-full border border-slate-200 bg-slate-50 rounded-xl p-3 text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
              >
                {results.map((r) => (
                  <option key={r.rank} value={r.rank}>
                    Rank {r.rank}: {r.candidate_id} (Score: {r.score.toFixed(3)})
                  </option>
                ))}
              </select>
            </div>

            {/* Results Table */}
            <div className="overflow-x-auto border border-slate-100 rounded-xl">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-slate-50 border-b border-slate-100 text-slate-500 uppercase tracking-wider font-bold">
                    <th className="p-3 w-16 text-center">Rank</th>
                    <th className="p-3 w-32">Candidate ID</th>
                    <th className="p-3 w-20">Score</th>
                    <th className="p-3">Summary Reasoning</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {results.map((r) => (
                    <tr 
                      key={r.rank}
                      onClick={() => setSelectedRank(r.rank)}
                      className={`hover:bg-indigo-50/30 cursor-pointer transition ${r.rank === selectedRank ? 'bg-indigo-50/50' : ''}`}
                    >
                      <td className="p-3 text-center font-bold text-slate-500">{r.rank}</td>
                      <td className="p-3 font-semibold text-slate-700">{r.candidate_id}</td>
                      <td className="p-3 font-mono font-bold text-indigo-600">{r.score.toFixed(3)}</td>
                      <td className="p-3 text-slate-500 max-w-[200px] truncate">{r.reasoning}</td>
                    </tr>
                  ))}
                  {results.length === 0 && (
                    <tr>
                      <td colSpan={4} className="p-8 text-center text-slate-400">
                        {isApiLoading ? "Loading candidate results..." : "No candidates found."}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Right: Selected Candidate Profile */}
          <div className="flex flex-col gap-6">
            
            {/* Header info */}
            {selectedCandidate ? (
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 border-l-4 border-l-blue-600 flex flex-col gap-3">
                <div>
                  <h2 className="text-xl font-bold text-slate-900">
                    Anonymous Candidate
                  </h2>
                  <div className="text-sm font-semibold text-indigo-600 mt-1">
                    {selectedCandidate.candidate_id}
                  </div>
                </div>
                <div className="text-xs text-slate-500 italic mt-1 leading-relaxed font-semibold">
                  "{selectedCandidate.reasoning}"
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 text-center text-slate-400 text-sm font-semibold">
                Select a candidate to view details
              </div>
            )}

            {/* Detailed profile tabs */}
            {selectedCandidate && (
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 flex flex-col gap-4">
                
                {/* Tabs bar */}
                <div className="flex border-b border-slate-100 gap-4 overflow-x-auto text-xs font-bold">
                  {[
                    { id: "score", label: "📊 Score Breakdown" },
                    { id: "naive", label: "⚡ vs Naive" },
                    { id: "contrast", label: "🔍 Contrast Cases" }
                  ].map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setSelectedTab(tab.id)}
                      className={`pb-3 border-b-2 transition focus:outline-none whitespace-nowrap ${selectedTab === tab.id ? 'border-indigo-600 text-indigo-600' : 'border-transparent text-slate-400 hover:text-slate-600'}`}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>

                {/* Tab content: 1. Score Breakdown */}
                {selectedTab === "score" && (
                  <div className="flex flex-col gap-4">
                    
                    {/* Fit Tier details */}
                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div className="p-3 bg-slate-50 rounded-xl border border-slate-100 border-l-4 border-l-indigo-600 border-t border-r border-b">
                        <span className="text-slate-400 font-semibold block mb-0.5">Assigned Tier:</span>
                        <span className="font-extrabold text-indigo-600">{selectedCandidate.tier_label || "Good Fit"}</span>
                      </div>
                      <div className="p-3 bg-slate-50 rounded-xl border border-slate-100 border-l-4 border-l-slate-600 border-t border-r border-b">
                        <span className="text-slate-400 font-semibold block mb-0.5">Confidence Level:</span>
                        <span className="font-extrabold text-slate-900">{selectedCandidate.confidence || "high"}</span>
                      </div>
                    </div>

                    <h4 className="font-bold text-xs text-slate-700 uppercase tracking-wider mt-2 font-semibold">Dimension Breakdown:</h4>
                    <div className="flex flex-col gap-3">
                      {Object.keys(selectedCandidate.score_card || {}).map((key) => {
                        const dim = selectedCandidate.score_card[key];
                        return (
                          <div key={key} className="flex flex-col gap-1 border-b border-slate-50 pb-2">
                            <div className="flex items-center justify-between text-xs font-semibold">
                              <span className="text-slate-700">{dim.label}</span>
                              <div>
                                <span className="text-sky-600 font-bold">{dim.score.toFixed(4)}</span>
                                <span className="text-slate-400 font-normal text-[10px]"> ×{dim.weight.toFixed(3)} = {dim.contribution.toFixed(4)}</span>
                              </div>
                            </div>
                            <div className="text-[10px] text-slate-500 bg-slate-50 px-2 py-1 border border-slate-100 rounded">
                              {dim.evidence}
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    {/* Computation card */}
                    <div className="bg-indigo-50 border border-indigo-100 rounded-xl p-4 flex flex-col gap-2 mt-2">
                      <div className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider">Score Computation:</div>
                      <div className="flex justify-between text-xs text-slate-700">
                        <span>Base weighted total:</span>
                        <span className="font-mono font-semibold">{selectedCandidate.weighted_total.toFixed(4)}</span>
                      </div>
                      <div className="flex justify-between text-xs text-slate-700">
                        <span>Behavioral multiplier:</span>
                        <span className="font-mono font-semibold text-indigo-600">×{selectedCandidate.behavioral?.value?.toFixed(4) || "1.0000"}</span>
                      </div>
                      <div className="text-[10px] text-slate-500 italic leading-relaxed font-semibold">
                        {selectedCandidate.behavioral?.evidence}
                      </div>
                      <div className="flex justify-between text-sm font-extrabold border-t border-indigo-200/50 pt-2 text-slate-800">
                        <span>Composite Score:</span>
                        <span className="text-blue-700">{selectedCandidate.score.toFixed(4)}</span>
                      </div>
                    </div>

                    {/* Concerns card */}
                    {selectedCandidate.why_not_notes && selectedCandidate.why_not_notes.length > 0 && (
                      <div className="flex flex-col gap-2 mt-2">
                        <h4 className="font-bold text-xs text-slate-700 uppercase tracking-wider">⚠️ Identified Concerns:</h4>
                        {selectedCandidate.why_not_notes.map((note, idx) => (
                          <div key={idx} className="text-xs text-amber-700 bg-amber-50/70 border border-amber-200 px-3 py-2 rounded-lg flex gap-2 items-center font-semibold">
                            <span className="text-amber-500">⚠</span>
                            <span>{note}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Tab content: 2. Naive compare */}
                {selectedTab === "naive" && (
                  <div className="flex flex-col gap-4 text-xs">
                    <p className="text-slate-500 leading-relaxed font-semibold">
                      A naive keyword matcher only counts word occurrences. Our pipeline accounts for pedigree, location, coherence, and behavioral data.
                    </p>
                    
                    <div className="grid grid-cols-2 gap-4 my-2">
                      <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl text-center">
                        <div className="text-slate-400 font-bold uppercase tracking-wider text-[10px] mb-1">Naive Keyword Rank</div>
                        <div className="text-2xl font-black text-slate-500">#{selectedCandidate.naive_rank || "N/A"}</div>
                      </div>
                      <div className="p-4 bg-indigo-50 border border-indigo-100 rounded-xl text-center">
                        <div className="text-indigo-600 font-bold uppercase tracking-wider text-[10px] mb-1">Pipeline Rank</div>
                        <div className="text-2xl font-black text-indigo-700">#{selectedCandidate.rank}</div>
                      </div>
                    </div>

                    {selectedCandidate.naive_rank_change !== null && (
                      <div className={`p-3 border rounded-xl flex items-center gap-2 font-bold text-sm ${selectedCandidate.naive_rank_change >= 0 ? 'bg-emerald-50 border-emerald-200 text-emerald-700' : 'bg-rose-50 border-rose-200 text-rose-700'}`}>
                        {selectedCandidate.naive_rank_change >= 0 ? (
                          <>
                            <ArrowUpRight size={18} />
                            <span>Promoted by {Math.abs(selectedCandidate.naive_rank_change)} places</span>
                          </>
                        ) : (
                          <>
                            <TrendingDown size={18} />
                            <span>Demoted by {Math.abs(selectedCandidate.naive_rank_change)} places</span>
                          </>
                        )}
                      </div>
                    )}

                    <h4 className="font-bold text-xs text-slate-700 uppercase tracking-wider mt-2 font-semibold">Top 3 Pipeline vs Naive Rankings:</h4>
                    <div className="flex flex-col border border-slate-100 rounded-xl divide-y divide-slate-100">
                      <div className="grid grid-cols-3 p-2 bg-slate-50 font-bold text-[10px] uppercase text-slate-500 tracking-wider">
                        <span>Candidate</span>
                        <span className="text-center">Naive Rank</span>
                        <span className="text-center">Pipeline Rank</span>
                      </div>
                      {naiveCompare.pipeline_top20?.slice(0, 3).map((item, idx) => (
                        <div key={idx} className="grid grid-cols-3 p-2 items-center">
                          <span className="font-semibold text-slate-700">{item.candidate_id}</span>
                          <span className="text-center font-semibold text-slate-400">#{item.naive_rank}</span>
                          <span className="text-center font-bold text-indigo-600">#{item.pipeline_rank}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Tab content: 3. Contrast cases */}
                {selectedTab === "contrast" && (
                  <div className="flex flex-col gap-4 text-xs">
                    <p className="text-slate-500 leading-relaxed mb-2 font-semibold">
                      Proof of robustness: profiles that keyword-match perfectly but fail on coherence or timeline tests.
                    </p>

                    {/* Honeypot card */}
                    <div className="flex flex-col gap-2">
                      <h4 className="font-bold text-xs text-slate-700 uppercase tracking-wider">🛡️ Rejected Honeypot Example:</h4>
                      {contrastData.honeypot_cards && contrastData.honeypot_cards.length > 0 ? (
                        <div className="bg-rose-50/50 border border-rose-200 rounded-xl p-4 flex flex-col gap-2">
                          <div className="font-bold text-slate-800">
                            {contrastData.honeypot_cards[0].title || "Frontend Developer"} @ {contrastData.honeypot_cards[0].company || "Mock Corp"}
                          </div>
                          <div className="font-semibold text-rose-600 flex items-center gap-1.5">
                            🚫 {contrastData.honeypot_cards[0].outcome}
                          </div>
                          <div className="flex flex-col gap-1.5 mt-1">
                            {contrastData.honeypot_cards[0].flag_explanations?.map((flag, idx) => (
                              <div key={idx} className="bg-white border border-rose-100 text-rose-700 font-semibold p-2 border-l-2 border-l-rose-500 rounded">
                                {flag}
                              </div>
                            ))}
                          </div>
                          <div className="text-[10px] text-slate-400 italic mt-1 leading-normal">
                            {contrastData.honeypot_cards[0].pipeline_note}
                          </div>
                        </div>
                      ) : (
                        <div className="text-slate-400 italic text-center p-4 border border-slate-100 rounded-xl">
                          No honeypot examples in current dataset.
                        </div>
                      )}
                    </div>

                    {/* Demotion card */}
                    <div className="flex flex-col gap-2 mt-2">
                      <h4 className="font-bold text-xs text-slate-700 uppercase tracking-wider">📉 Keyword-Stuffer Demoted:</h4>
                      {contrastData.demotion_cards && contrastData.demotion_cards.length > 0 ? (
                        <div className="bg-amber-50/40 border border-amber-200 rounded-xl p-4 flex flex-col gap-2">
                          <div className="font-bold text-slate-800">
                            {contrastData.demotion_cards[0].title} @ {contrastData.demotion_cards[0].company}
                          </div>
                          <div className="font-semibold text-amber-600 flex items-center gap-1.5">
                            ⚠ {contrastData.demotion_cards[0].outcome}
                          </div>
                          
                          <div className="grid grid-cols-2 gap-3 my-1">
                            <div className="bg-white border border-slate-200 rounded-lg p-2 text-center">
                              <span className="text-[10px] text-slate-400 block font-semibold">Naive Keyword Rank</span>
                              <span className="text-sm font-bold text-slate-400">#{contrastData.demotion_cards[0].naive_rank}</span>
                            </div>
                            <div className="bg-indigo-50 border border-indigo-100 rounded-lg p-2 text-center">
                              <span className="text-[10px] text-indigo-600 block font-semibold">Pipeline Rank</span>
                              <span className="text-sm font-bold text-indigo-600">#{contrastData.demotion_cards[0].pipeline_rank}</span>
                            </div>
                          </div>

                          <p className="text-[11px] text-slate-600 leading-relaxed font-semibold bg-white p-2 border border-amber-100 rounded">
                            {contrastData.demotion_cards[0].demotion_reason}
                          </p>

                          <div className="text-[10px] text-slate-400 leading-normal">
                            JD-skill hits: {contrastData.demotion_cards[0].jd_skill_hits}/{contrastData.demotion_cards[0].total_skills} skills | Coherence multiplier: ×{contrastData.demotion_cards[0].coherence_multiplier}
                          </div>
                        </div>
                      ) : (
                        <div className="text-slate-400 italic text-center p-4 border border-slate-100 rounded-xl">
                          No demotion examples in current dataset.
                        </div>
                      )}
                    </div>
                  </div>
                )}

              </div>
            )}
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
