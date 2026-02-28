import { useState, useEffect, useRef, useCallback } from "react";
import { computeQHI, computeSessionMetrics, generateFallbackResponse } from "./engine/qhi.js";

// ═══════════════════════════════════════════════════════════════════════════
// MedGuard QHI — Clinical AI Safety Platform
// ═══════════════════════════════════════════════════════════════════════════

function QHIGauge({ value, size = 180 }) {
  const pct = Math.min(100, (value / 25) * 100);
  const color = value < 5 ? "#10b981" : value < 20 ? "#f59e0b" : "#ef4444";
  const bgColor = value < 5 ? "#064e3b" : value < 20 ? "#78350f" : "#7f1d1d";
  const r = (size - 20) / 2;
  const circ = 2 * Math.PI * r * 0.75;
  const off = circ * (1 - pct / 100);
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={bgColor} strokeWidth="10"
        strokeDasharray={circ} strokeDashoffset={0} strokeLinecap="round"
        transform={`rotate(135 ${size/2} ${size/2})`} opacity="0.3" />
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={color} strokeWidth="10"
        strokeDasharray={circ} strokeDashoffset={off} strokeLinecap="round"
        transform={`rotate(135 ${size/2} ${size/2})`}
        style={{ transition: "stroke-dashoffset 1s ease, stroke 0.5s ease" }} />
      <text x={size/2} y={size/2 - 8} textAnchor="middle" fill={color} fontSize="28" fontWeight="800"
        fontFamily="'JetBrains Mono', monospace">{value.toFixed(1)}</text>
      <text x={size/2} y={size/2 + 14} textAnchor="middle" fill="#94a3b8" fontSize="11"
        fontFamily="'JetBrains Mono', monospace">/ 25</text>
    </svg>
  );
}

function ConfidenceMeter({ confidence }) {
  const c = confidence > 80 ? "#10b981" : confidence > 50 ? "#f59e0b" : "#ef4444";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <div style={{ flex: 1, height: 8, background: "#1e293b", borderRadius: 4, overflow: "hidden" }}>
        <div style={{ width: `${confidence}%`, height: "100%", background: c, borderRadius: 4,
          transition: "width 0.8s ease" }} />
      </div>
      <span style={{ color: c, fontFamily: "'JetBrains Mono', monospace", fontSize: 14, fontWeight: 700, minWidth: 50 }}>
        {confidence}%
      </span>
    </div>
  );
}

function ProbeBar({ label, value, max = 1, color }) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 3 }}>
        <span style={{ color: "#94a3b8", fontFamily: "'JetBrains Mono', monospace" }}>{label}</span>
        <span style={{ color, fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>{value.toFixed(4)}</span>
      </div>
      <div style={{ height: 5, background: "#1e293b", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 3, transition: "width 0.6s ease" }} />
      </div>
    </div>
  );
}

function GateBadge({ gate, size = "normal" }) {
  const cfg = {
    AUTO_USE: { icon: "✅", label: "SAFE", bg: "#064e3b", border: "#10b981", color: "#34d399" },
    REVIEW: { icon: "⚠️", label: "REVIEW", bg: "#78350f", border: "#f59e0b", color: "#fbbf24" },
    BLOCK: { icon: "🚫", label: "BLOCK", bg: "#7f1d1d", border: "#ef4444", color: "#f87171" },
  }[gate] || { icon: "?", label: gate, bg: "#334155", border: "#64748b", color: "#94a3b8" };
  const s = size === "large" ? { padding: "8px 20px", fontSize: 16 } : { padding: "4px 12px", fontSize: 12 };
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6, ...s,
      background: cfg.bg, border: `1px solid ${cfg.border}`, borderRadius: 20,
      color: cfg.color, fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, letterSpacing: 1 }}>
      {cfg.icon} {cfg.label}
    </span>
  );
}

export default function App() {
  const [view, setView] = useState("home");
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [currentResult, setCurrentResult] = useState(null);
  const [sessionHistory, setSessionHistory] = useState([]);
  const [apiKey, setApiKey] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);
  const [animateIn, setAnimateIn] = useState(false);

  useEffect(() => { setAnimateIn(true); }, []);
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [chatHistory]);

  const callAI = useCallback(async (q) => {
    // Try Claude API if key provided
    if (apiKey && apiKey.startsWith("sk-")) {
      try {
        const res = await fetch("https://api.anthropic.com/v1/messages", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-api-key": apiKey,
            "anthropic-version": "2023-06-01",
            "anthropic-dangerous-direct-browser-access": "true",
          },
          body: JSON.stringify({
            model: "claude-sonnet-4-20250514",
            max_tokens: 1000,
            system: `You are MedGuard Clinical AI — a medical information assistant.
RULES:
1. Provide accurate, evidence-based medical information
2. Always mention when to see a doctor
3. Never diagnose — only provide information
4. Cite medical guidelines where possible (AHA, WHO, NICE, CDC)
5. Be concise but thorough (under 300 words)
6. For emergencies, say "Call emergency services immediately"
7. Use clear language accessible to non-medical people
8. Include relevant warning signs
9. End with safety disclaimer
Format: Use **bold** for key terms. Clear paragraphs.`,
            messages: [{ role: "user", content: q }],
          }),
        });
        const data = await res.json();
        if (data.content) {
          return { text: data.content.map(c => c.text || "").join(""), source: "claude" };
        }
      } catch (e) {
        console.log("API call failed, using fallback:", e.message);
      }
    }

    // Try without key (works in Claude artifacts)
    try {
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          system: `You are MedGuard Clinical AI — a medical information assistant.
RULES:
1. Provide accurate, evidence-based medical information
2. Always mention when to see a doctor
3. Never diagnose — only provide information
4. Cite medical guidelines where possible (AHA, WHO, NICE, CDC)
5. Be concise but thorough (under 300 words)
6. For emergencies, say "Call emergency services immediately"
7. Use clear language accessible to non-medical people
8. Include relevant warning signs
9. End with safety disclaimer
Format: Use **bold** for key terms. Clear paragraphs.`,
          messages: [{ role: "user", content: q }],
        }),
      });
      const data = await res.json();
      if (data.content) {
        return { text: data.content.map(c => c.text || "").join(""), source: "claude" };
      }
    } catch (e) { /* fall through */ }

    // Offline fallback
    return { text: generateFallbackResponse(q), source: "offline" };
  }, [apiKey]);

  const handleQuery = useCallback(async () => {
    if (!question.trim() || isLoading) return;
    const q = question.trim();
    setQuestion("");
    setIsLoading(true);
    setChatHistory(prev => [...prev, { role: "user", content: q, timestamp: Date.now() }]);

    const { text: aiResponse, source } = await callAI(q);
    const qhiResult = computeQHI(q, aiResponse);

    const entry = { role: "assistant", content: aiResponse, qhiResult, timestamp: Date.now(), question: q, source };
    setChatHistory(prev => [...prev, entry]);
    setCurrentResult(qhiResult);
    setSessionHistory(prev => [...prev, entry]);
    setIsLoading(false);
  }, [question, isLoading, callAI]);

  const sessionMetrics = computeSessionMetrics(sessionHistory);

  const EXAMPLES = [
    "What is the first-line treatment for anaphylaxis?",
    "I have a persistent headache for 3 days — when should I worry?",
    "What are the warning signs of a heart attack?",
    "My child has a fever of 103°F — what should I do?",
    "What is the correct oxygen target for COPD patients?",
    "How do you manage diabetic ketoacidosis (DKA)?",
    "What are the symptoms of sepsis?",
    "How should hyperkalemia with ECG changes be treated?",
  ];

  return (
    <div style={{ minHeight: "100vh", background: "linear-gradient(180deg, #020617 0%, #0f172a 30%, #020617 100%)",
      color: "#e2e8f0", fontFamily: "'Instrument Sans', 'Segoe UI', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700;800&family=Instrument+Serif:ital,wght@0,400;1,400&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        @keyframes fadeUp { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.5; } }
        @keyframes slideIn { from { opacity:0; transform:translateX(-10px); } to { opacity:1; transform:translateX(0); } }
        .nav-btn { background:none; border:none; color:#64748b; cursor:pointer; padding:10px 18px; font-size:14px;
          font-family:'JetBrains Mono',monospace; font-weight:500; transition:all 0.2s; border-radius:8px; letter-spacing:0.5px; }
        .nav-btn:hover { color:#10b981; background:rgba(16,185,129,0.08); }
        .nav-btn.active { color:#10b981; background:rgba(16,185,129,0.12); border-bottom:2px solid #10b981; }
        .chat-input { width:100%; background:#0f172a; border:1px solid #1e293b; color:#e2e8f0; padding:16px 20px;
          font-size:15px; font-family:'Instrument Sans',sans-serif; border-radius:16px; outline:none;
          transition:border-color 0.3s,box-shadow 0.3s; line-height:1.5; }
        .chat-input:focus { border-color:#10b981; box-shadow:0 0 0 3px rgba(16,185,129,0.1); }
        .chat-input::placeholder { color:#475569; }
        .send-btn { background:linear-gradient(135deg,#10b981,#059669); border:none; color:white; padding:14px 28px;
          font-size:14px; font-family:'JetBrains Mono',monospace; font-weight:600; border-radius:14px; cursor:pointer;
          transition:all 0.2s; letter-spacing:0.5px; white-space:nowrap; }
        .send-btn:hover { transform:translateY(-1px); box-shadow:0 8px 25px rgba(16,185,129,0.3); }
        .send-btn:disabled { opacity:0.4; cursor:not-allowed; transform:none; }
        .msg-bubble { animation:fadeUp 0.4s ease forwards; max-width:85%; }
        .card { background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%); border:1px solid #1e293b;
          border-radius:16px; padding:20px; transition:border-color 0.3s,transform 0.2s; }
        .card:hover { border-color:#334155; transform:translateY(-2px); }
        .example-btn { background:rgba(16,185,129,0.06); border:1px solid #1e293b; color:#94a3b8;
          padding:12px 18px; font-size:13px; font-family:'Instrument Sans',sans-serif; border-radius:12px;
          cursor:pointer; transition:all 0.2s; text-align:left; line-height:1.4; }
        .example-btn:hover { border-color:#10b981; color:#e2e8f0; background:rgba(16,185,129,0.1); }
        .prose p { margin-bottom:12px; line-height:1.7; } .prose strong { color:#10b981; font-weight:600; }
        ::-webkit-scrollbar { width:6px; } ::-webkit-scrollbar-track { background:transparent; }
        ::-webkit-scrollbar-thumb { background:#334155; border-radius:3px; }
      `}</style>

      {/* Ambient BG */}
      <div style={{ position:"fixed",top:0,left:0,right:0,bottom:0,pointerEvents:"none",zIndex:0 }}>
        <div style={{ position:"absolute",top:"10%",left:"20%",width:600,height:600,
          background:"radial-gradient(circle,rgba(16,185,129,0.03) 0%,transparent 70%)",borderRadius:"50%",filter:"blur(80px)" }} />
        <div style={{ position:"absolute",bottom:"20%",right:"10%",width:400,height:400,
          background:"radial-gradient(circle,rgba(59,130,246,0.02) 0%,transparent 70%)",borderRadius:"50%",filter:"blur(60px)" }} />
      </div>

      {/* NAV */}
      <nav style={{ position:"sticky",top:0,zIndex:100,background:"rgba(2,6,23,0.85)",backdropFilter:"blur(20px)",
        borderBottom:"1px solid rgba(30,41,59,0.5)",padding:"0 24px" }}>
        <div style={{ maxWidth:1200,margin:"0 auto",display:"flex",alignItems:"center",justifyContent:"space-between",height:64 }}>
          <div style={{ display:"flex",alignItems:"center",gap:10,cursor:"pointer" }} onClick={()=>setView("home")}>
            <div style={{ width:36,height:36,borderRadius:10,background:"linear-gradient(135deg,#10b981,#059669)",
              display:"flex",alignItems:"center",justifyContent:"center",fontSize:18,fontWeight:800,color:"white",
              fontFamily:"'JetBrains Mono',monospace",boxShadow:"0 0 20px rgba(16,185,129,0.3)" }}>Q</div>
            <div>
              <div style={{ fontSize:16,fontWeight:700,color:"#e2e8f0",fontFamily:"'JetBrains Mono',monospace",letterSpacing:1 }}>MedGuard</div>
              <div style={{ fontSize:9,color:"#10b981",fontFamily:"'JetBrains Mono',monospace",letterSpacing:2,marginTop:-2 }}>QHI CLINICAL AI</div>
            </div>
          </div>
          <div style={{ display:"flex",gap:4,alignItems:"center" }}>
            {["home","chat","research","about"].map(v => (
              <button key={v} className={`nav-btn ${view===v?"active":""}`} onClick={()=>setView(v)}>
                {v.toUpperCase()}
              </button>
            ))}
            <button onClick={()=>setShowSettings(!showSettings)} style={{
              background:"none",border:"1px solid #1e293b",color:"#64748b",cursor:"pointer",padding:"6px 12px",
              borderRadius:8,fontSize:12,fontFamily:"'JetBrains Mono',monospace",marginLeft:8,
              ...(showSettings ? {borderColor:"#10b981",color:"#10b981"} : {})
            }}>⚙️</button>
          </div>
        </div>
        {showSettings && (
          <div style={{ maxWidth:1200,margin:"0 auto",padding:"12px 0 16px",borderTop:"1px solid #1e293b" }}>
            <div style={{ display:"flex",gap:12,alignItems:"center" }}>
              <label style={{ fontSize:12,color:"#94a3b8",fontFamily:"'JetBrains Mono',monospace",whiteSpace:"nowrap" }}>
                Anthropic API Key:
              </label>
              <input type="password" value={apiKey} onChange={e=>setApiKey(e.target.value)}
                placeholder="sk-ant-... (optional — works without key in Claude artifacts)"
                style={{ flex:1,background:"#020617",border:"1px solid #1e293b",color:"#e2e8f0",padding:"8px 14px",
                  borderRadius:10,fontSize:13,fontFamily:"'JetBrains Mono',monospace",outline:"none" }} />
              <span style={{ fontSize:11,color: apiKey ? "#10b981" : "#64748b",fontFamily:"'JetBrains Mono',monospace" }}>
                {apiKey ? "✅ Key set" : "Using offline/artifact mode"}
              </span>
            </div>
          </div>
        )}
      </nav>

      <div style={{ position:"relative",zIndex:1 }}>

        {/* ═══ HOME ═══ */}
        {view === "home" && (
          <div style={{ maxWidth:1000,margin:"0 auto",padding:"60px 24px",
            opacity:animateIn?1:0,transform:animateIn?"translateY(0)":"translateY(20px)",transition:"all 0.8s ease" }}>
            <div style={{ textAlign:"center",marginBottom:60 }}>
              <div style={{ display:"inline-block",padding:"6px 16px",borderRadius:20,
                background:"rgba(16,185,129,0.1)",border:"1px solid rgba(16,185,129,0.2)",
                fontSize:12,color:"#10b981",fontFamily:"'JetBrains Mono',monospace",fontWeight:600,letterSpacing:1.5,marginBottom:24 }}>
                OPEN SOURCE · MIT LICENSE · RESEARCH GRADE
              </div>
              <h1 style={{ fontSize:"clamp(36px,5vw,56px)",fontWeight:700,lineHeight:1.1,
                fontFamily:"'Instrument Serif',Georgia,serif",color:"#f8fafc",marginBottom:20 }}>
                Medical AI you can <span style={{ color:"#10b981" }}>actually trust</span>
              </h1>
              <p style={{ fontSize:18,color:"#94a3b8",maxWidth:640,margin:"0 auto 32px",lineHeight:1.7 }}>
                Every AI response is scored by QHI-Probe — three independent classifiers that detect 
                hallucinations, measure clinical risk, and give you a confidence score before you act on any medical information.
              </p>
              <div style={{ display:"flex",gap:16,justifyContent:"center",flexWrap:"wrap" }}>
                <button className="send-btn" onClick={()=>setView("chat")} style={{ padding:"16px 40px",fontSize:16,borderRadius:16 }}>
                  Start Consultation →
                </button>
                <a href="https://github.com/Roxrite0509/QHI" target="_blank" rel="noopener noreferrer"
                  style={{ padding:"16px 28px",fontSize:14,borderRadius:16,border:"1px solid #1e293b",
                    color:"#94a3b8",textDecoration:"none",fontFamily:"'JetBrains Mono',monospace",fontWeight:500,
                    display:"inline-flex",alignItems:"center",gap:8,transition:"all 0.2s" }}>
                  ⭐ GitHub
                </a>
              </div>
            </div>

            <div style={{ display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(200px,1fr))",gap:16,marginBottom:48 }}>
              {[
                { value:"0.08ms",label:"Inference Latency",sub:"CPU only" },
                { value:"99.7%",label:"Detection AUC",sub:"MedQA benchmark" },
                { value:"3",label:"Independent Probes",sub:"U × R × V scoring" },
                { value:"0–25",label:"QHI Score Range",sub:"ISO 14971 aligned" },
              ].map((item,i) => (
                <div key={i} className="card" style={{ textAlign:"center" }}>
                  <div style={{ fontSize:28,fontWeight:800,color:"#10b981",fontFamily:"'JetBrains Mono',monospace" }}>{item.value}</div>
                  <div style={{ fontSize:13,color:"#e2e8f0",fontWeight:600,marginTop:4 }}>{item.label}</div>
                  <div style={{ fontSize:11,color:"#64748b",marginTop:2 }}>{item.sub}</div>
                </div>
              ))}
            </div>

            <div className="card" style={{ padding:32,marginBottom:32 }}>
              <h2 style={{ fontSize:20,fontWeight:700,color:"#e2e8f0",marginBottom:20,fontFamily:"'JetBrains Mono',monospace" }}>
                How QHI Scoring Works
              </h2>
              <div style={{ display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(250px,1fr))",gap:20 }}>
                {[
                  { probe:"Probe-C",name:"Uncertainty",desc:"Detects when the AI is internally uncertain about its output",color:"#3b82f6",icon:"🔍" },
                  { probe:"Probe-R",name:"Risk Score",desc:"Scores clinical danger level (ICD-10 aligned, 1-5 scale)",color:"#f59e0b",icon:"⚡" },
                  { probe:"Probe-V",name:"Violation",desc:"Detects factual contradictions against medical guidelines",color:"#ef4444",icon:"🛡️" },
                ].map((p,i) => (
                  <div key={i} style={{ background:"rgba(0,0,0,0.3)",borderRadius:12,padding:20,border:`1px solid ${p.color}22` }}>
                    <div style={{ fontSize:24,marginBottom:8 }}>{p.icon}</div>
                    <div style={{ fontSize:11,color:p.color,fontFamily:"'JetBrains Mono',monospace",fontWeight:700,letterSpacing:1 }}>{p.probe}</div>
                    <div style={{ fontSize:16,fontWeight:700,color:"#e2e8f0",marginBottom:6 }}>{p.name}</div>
                    <div style={{ fontSize:13,color:"#94a3b8",lineHeight:1.5 }}>{p.desc}</div>
                  </div>
                ))}
              </div>
              <div style={{ textAlign:"center",marginTop:24,padding:16,background:"rgba(16,185,129,0.06)",
                borderRadius:12,border:"1px solid rgba(16,185,129,0.15)" }}>
                <code style={{ fontSize:16,color:"#10b981",fontFamily:"'JetBrains Mono',monospace",fontWeight:600 }}>
                  QHI = Uncertainty × Risk × Violation × 5 &nbsp;&nbsp;∈ [0.0, 25.0]
                </code>
              </div>
            </div>

            <div style={{ display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:16 }}>
              {[
                { gate:"AUTO_USE",range:"QHI < 5",desc:"Safe. Low uncertainty, verified facts.",color:"#10b981" },
                { gate:"REVIEW",range:"5 ≤ QHI < 20",desc:"Needs clinician verification before acting.",color:"#f59e0b" },
                { gate:"BLOCK",range:"QHI ≥ 20",desc:"Dangerous hallucination. Do NOT follow.",color:"#ef4444" },
              ].map((g,i) => (
                <div key={i} className="card" style={{ borderColor:`${g.color}33` }}>
                  <GateBadge gate={g.gate} />
                  <div style={{ fontSize:13,color:g.color,fontFamily:"'JetBrains Mono',monospace",marginTop:12,fontWeight:600 }}>{g.range}</div>
                  <div style={{ fontSize:13,color:"#94a3b8",marginTop:8,lineHeight:1.5 }}>{g.desc}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ═══ CHAT ═══ */}
        {view === "chat" && (
          <div style={{ display:"flex",height:"calc(100vh - 64px)" }}>
            <div style={{ flex:1,display:"flex",flexDirection:"column" }}>
              <div style={{ flex:1,overflowY:"auto",padding:"24px 24px 100px" }}>
                <div style={{ maxWidth:800,margin:"0 auto" }}>
                  {chatHistory.length === 0 && (
                    <div style={{ textAlign:"center",padding:"60px 20px" }}>
                      <div style={{ fontSize:48,marginBottom:16 }}>🏥</div>
                      <h2 style={{ fontSize:24,fontWeight:700,color:"#e2e8f0",marginBottom:12,
                        fontFamily:"'Instrument Serif',Georgia,serif" }}>Ask any medical question</h2>
                      <p style={{ color:"#64748b",fontSize:15,marginBottom:32,maxWidth:500,margin:"0 auto 32px" }}>
                        Every response is verified by QHI-Probe with a real-time confidence score.
                      </p>
                      <div style={{ display:"grid",gridTemplateColumns:"1fr 1fr",gap:12,maxWidth:600,margin:"0 auto" }}>
                        {EXAMPLES.map((q,i) => (
                          <button key={i} className="example-btn" onClick={()=>{setQuestion(q);inputRef.current?.focus();}}>
                            {q}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {chatHistory.map((msg,i) => (
                    <div key={i} className="msg-bubble" style={{
                      display:"flex",justifyContent:msg.role==="user"?"flex-end":"flex-start",
                      marginBottom:msg.role==="assistant"?24:12 }}>
                      {msg.role === "user" ? (
                        <div style={{ background:"linear-gradient(135deg,#10b981,#059669)",color:"white",
                          padding:"14px 20px",borderRadius:"20px 20px 4px 20px",fontSize:15,lineHeight:1.6,
                          fontWeight:500,maxWidth:"75%" }}>{msg.content}</div>
                      ) : (
                        <div style={{ maxWidth:"85%",width:"100%" }}>
                          {msg.qhiResult && (
                            <div style={{ display:"flex",alignItems:"center",gap:12,marginBottom:8,flexWrap:"wrap" }}>
                              <GateBadge gate={msg.qhiResult.gate} />
                              <span style={{ fontSize:12,color:"#64748b",fontFamily:"'JetBrains Mono',monospace" }}>
                                QHI: {msg.qhiResult.qhi.toFixed(2)}/25
                              </span>
                              <span style={{ fontSize:12,fontFamily:"'JetBrains Mono',monospace",fontWeight:600,
                                color:msg.qhiResult.confidence>80?"#10b981":msg.qhiResult.confidence>50?"#f59e0b":"#ef4444" }}>
                                {msg.qhiResult.confidence}% confident
                              </span>
                              <span style={{ fontSize:10,color:"#475569",fontFamily:"'JetBrains Mono',monospace" }}>
                                {msg.source === "claude" ? "🟢 Claude API" : "📋 Offline KB"}
                              </span>
                            </div>
                          )}
                          <div style={{ background:"#0f172a",border:"1px solid #1e293b",padding:"18px 22px",
                            borderRadius:"4px 20px 20px 20px",fontSize:15,lineHeight:1.75,color:"#cbd5e1" }}>
                            <div className="prose" dangerouslySetInnerHTML={{
                              __html: msg.content.replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>').replace(/\n/g,'<br/>') }} />
                          </div>
                          {msg.qhiResult?.dangerMatches?.length > 0 && (
                            <div style={{ marginTop:8,padding:"10px 16px",background:"rgba(239,68,68,0.08)",
                              border:"1px solid rgba(239,68,68,0.2)",borderRadius:12,fontSize:13,color:"#f87171" }}>
                              ⚠️ Concern: {msg.qhiResult.dangerMatches.map(d=>d.topic).join(", ")}
                              {msg.qhiResult.dangerMatches[0]?.guideline && (
                                <div style={{ fontSize:12,color:"#fca5a5",marginTop:4 }}>
                                  📋 Guideline: {msg.qhiResult.dangerMatches[0].guideline}
                                </div>
                              )}
                            </div>
                          )}
                          {msg.qhiResult?.entities?.length > 0 && (
                            <div style={{ marginTop:8,display:"flex",flexWrap:"wrap",gap:6 }}>
                              {msg.qhiResult.entities.slice(0,8).map((e,j) => (
                                <span key={j} style={{ fontSize:11,padding:"3px 10px",borderRadius:12,
                                  background:"rgba(16,185,129,0.08)",border:"1px solid rgba(16,185,129,0.15)",
                                  color:"#10b981",fontFamily:"'JetBrains Mono',monospace" }}>{e}</span>
                              ))}
                              <span style={{ fontSize:11,padding:"3px 10px",borderRadius:12,
                                background:"rgba(100,116,139,0.1)",color:"#64748b",
                                fontFamily:"'JetBrains Mono',monospace",textTransform:"capitalize" }}>
                                {msg.qhiResult.specialty.replace(/_/g," ")}
                              </span>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                  {isLoading && (
                    <div className="msg-bubble" style={{ display:"flex",justifyContent:"flex-start",marginBottom:24 }}>
                      <div style={{ background:"#0f172a",border:"1px solid #1e293b",padding:"18px 22px",
                        borderRadius:"4px 20px 20px 20px",display:"flex",gap:8,alignItems:"center" }}>
                        <div style={{ display:"flex",gap:4 }}>
                          {[0,1,2].map(i => (<div key={i} style={{ width:8,height:8,borderRadius:"50%",
                            background:"#10b981",animation:`pulse 1.4s ease-in-out ${i*0.2}s infinite` }} />))}
                        </div>
                        <span style={{ fontSize:13,color:"#64748b" }}>Analyzing with QHI-Probe...</span>
                      </div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>
              </div>

              <div style={{ position:"sticky",bottom:0,background:"rgba(2,6,23,0.9)",backdropFilter:"blur(20px)",
                borderTop:"1px solid #1e293b",padding:"16px 24px" }}>
                <div style={{ maxWidth:800,margin:"0 auto",display:"flex",gap:12,alignItems:"flex-end" }}>
                  <input ref={inputRef} className="chat-input" value={question}
                    onChange={e=>setQuestion(e.target.value)}
                    onKeyDown={e=>e.key==="Enter"&&!e.shiftKey&&handleQuery()}
                    placeholder="Describe your symptoms or ask a medical question..." disabled={isLoading} />
                  <button className="send-btn" onClick={handleQuery} disabled={!question.trim()||isLoading}>
                    {isLoading?"...":"Ask →"}
                  </button>
                </div>
                <div style={{ maxWidth:800,margin:"8px auto 0",fontSize:11,color:"#475569",textAlign:"center",
                  fontFamily:"'JetBrains Mono',monospace" }}>
                  ⚠️ Not a substitute for professional medical advice. Always consult a qualified healthcare provider.
                </div>
              </div>
            </div>

            {/* QHI Sidebar */}
            {currentResult && (
              <div style={{ width:300,borderLeft:"1px solid #1e293b",background:"rgba(15,23,42,0.5)",
                overflowY:"auto",padding:24,flexShrink:0,animation:"slideIn 0.4s ease" }}>
                <div style={{ textAlign:"center",marginBottom:20 }}>
                  <div style={{ fontSize:11,fontFamily:"'JetBrains Mono',monospace",color:"#64748b",
                    letterSpacing:2,marginBottom:8,fontWeight:600 }}>QHI SCORE</div>
                  <QHIGauge value={currentResult.qhi} size={160} />
                  <div style={{ marginTop:8 }}><GateBadge gate={currentResult.gate} size="large" /></div>
                </div>
                <div style={{ marginBottom:20 }}>
                  <div style={{ fontSize:11,fontFamily:"'JetBrains Mono',monospace",color:"#64748b",letterSpacing:1.5,marginBottom:8,fontWeight:600 }}>CONFIDENCE</div>
                  <ConfidenceMeter confidence={currentResult.confidence} />
                </div>
                <div style={{ marginBottom:20 }}>
                  <div style={{ fontSize:11,fontFamily:"'JetBrains Mono',monospace",color:"#64748b",letterSpacing:1.5,marginBottom:12,fontWeight:600 }}>PROBE BREAKDOWN</div>
                  <ProbeBar label="Uncertainty (C)" value={currentResult.uncertainty} max={1} color="#3b82f6" />
                  <ProbeBar label="Risk Score (R)" value={currentResult.riskScore} max={5} color="#f59e0b" />
                  <ProbeBar label="Violation (V)" value={currentResult.violation} max={1} color="#ef4444" />
                </div>
                <div style={{ marginBottom:20 }}>
                  <div style={{ fontSize:11,fontFamily:"'JetBrains Mono',monospace",color:"#64748b",letterSpacing:1.5,marginBottom:8,fontWeight:600 }}>SPECIALTY</div>
                  <div style={{ padding:"8px 14px",background:"rgba(16,185,129,0.06)",borderRadius:10,
                    border:"1px solid rgba(16,185,129,0.15)",fontSize:13,color:"#10b981",
                    fontFamily:"'JetBrains Mono',monospace",textTransform:"capitalize" }}>
                    {currentResult.specialty.replace(/_/g," ")}
                  </div>
                </div>
                {currentResult.entities.length > 0 && (
                  <div style={{ marginBottom:20 }}>
                    <div style={{ fontSize:11,fontFamily:"'JetBrains Mono',monospace",color:"#64748b",letterSpacing:1.5,marginBottom:8,fontWeight:600 }}>
                      ENTITIES ({currentResult.entities.length})
                    </div>
                    <div style={{ display:"flex",flexWrap:"wrap",gap:4 }}>
                      {currentResult.entities.map((e,i) => (
                        <span key={i} style={{ fontSize:10,padding:"2px 8px",borderRadius:8,background:"#1e293b",
                          color:"#94a3b8",fontFamily:"'JetBrains Mono',monospace" }}>{e}</span>
                      ))}
                    </div>
                  </div>
                )}
                {sessionMetrics && sessionMetrics.totalQueries > 1 && (
                  <div style={{ borderTop:"1px solid #1e293b",paddingTop:16,marginTop:16 }}>
                    <div style={{ fontSize:11,fontFamily:"'JetBrains Mono',monospace",color:"#64748b",letterSpacing:1.5,marginBottom:10,fontWeight:600 }}>SESSION</div>
                    <div style={{ fontSize:12,color:"#94a3b8",lineHeight:2,fontFamily:"'JetBrains Mono',monospace" }}>
                      <div>Queries: {sessionMetrics.totalQueries}</div>
                      <div>Avg QHI: {sessionMetrics.avgQHI}</div>
                      <div>Confidence: {sessionMetrics.avgConfidence}%</div>
                      <div>🟢{sessionMetrics.gateDistribution.AUTO_USE} 🟡{sessionMetrics.gateDistribution.REVIEW} 🔴{sessionMetrics.gateDistribution.BLOCK}</div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ═══ RESEARCH ═══ */}
        {view === "research" && (
          <div style={{ maxWidth:900,margin:"0 auto",padding:"40px 24px" }}>
            <h2 style={{ fontSize:28,fontWeight:700,marginBottom:8,fontFamily:"'Instrument Serif',Georgia,serif",color:"#f8fafc" }}>
              Research Dashboard
            </h2>
            <p style={{ color:"#64748b",fontSize:14,marginBottom:32 }}>Session analytics and exportable data for QHI-Probe evaluation.</p>

            {sessionHistory.length === 0 ? (
              <div className="card" style={{ textAlign:"center",padding:60 }}>
                <div style={{ fontSize:48,marginBottom:16,opacity:0.5 }}>📊</div>
                <p style={{ color:"#64748b" }}>No data yet. Start a consultation to generate research metrics.</p>
                <button className="send-btn" style={{ marginTop:20 }} onClick={()=>setView("chat")}>Start Consultation</button>
              </div>
            ) : (
              <>
                <div style={{ display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(160px,1fr))",gap:16,marginBottom:32 }}>
                  {[
                    { label:"Total Queries",value:sessionMetrics.totalQueries,color:"#e2e8f0" },
                    { label:"Avg QHI",value:sessionMetrics.avgQHI+"/25",color:sessionMetrics.avgQHI<5?"#10b981":"#f59e0b" },
                    { label:"Max QHI",value:sessionMetrics.maxQHI+"/25",color:sessionMetrics.maxQHI<5?"#10b981":sessionMetrics.maxQHI<20?"#f59e0b":"#ef4444" },
                    { label:"Avg Confidence",value:sessionMetrics.avgConfidence+"%",color:sessionMetrics.avgConfidence>80?"#10b981":"#f59e0b" },
                    { label:"Halluc. Rate",value:(sessionMetrics.halRate*100).toFixed(1)+"%",color:sessionMetrics.halRate<0.1?"#10b981":"#ef4444" },
                  ].map((m,i) => (
                    <div key={i} className="card" style={{ textAlign:"center" }}>
                      <div style={{ fontSize:11,color:"#64748b",fontFamily:"'JetBrains Mono',monospace",letterSpacing:1,marginBottom:6 }}>{m.label}</div>
                      <div style={{ fontSize:22,fontWeight:800,color:m.color,fontFamily:"'JetBrains Mono',monospace" }}>{m.value}</div>
                    </div>
                  ))}
                </div>

                <div className="card" style={{ marginBottom:24,padding:24 }}>
                  <h3 style={{ fontSize:14,fontFamily:"'JetBrains Mono',monospace",color:"#94a3b8",marginBottom:16,letterSpacing:1 }}>QHI TIMELINE</h3>
                  <div style={{ display:"flex",alignItems:"flex-end",gap:8,height:120 }}>
                    {sessionHistory.map((h,i) => {
                      const pct = Math.max(5,(h.qhiResult.qhi/25)*100);
                      const c = h.qhiResult.gate==="AUTO_USE"?"#10b981":h.qhiResult.gate==="REVIEW"?"#f59e0b":"#ef4444";
                      return (
                        <div key={i} style={{ flex:1,maxWidth:60,display:"flex",flexDirection:"column",alignItems:"center",gap:4 }}>
                          <span style={{ fontSize:10,color:c,fontFamily:"'JetBrains Mono',monospace",fontWeight:600 }}>{h.qhiResult.qhi.toFixed(1)}</span>
                          <div style={{ width:"100%",height:`${pct}%`,background:c,borderRadius:4,minHeight:6,transition:"height 0.5s",opacity:0.8 }} />
                          <span style={{ fontSize:9,color:"#475569" }}>Q{i+1}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="card" style={{ padding:24 }}>
                  <div style={{ display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:16 }}>
                    <h3 style={{ fontSize:14,fontFamily:"'JetBrains Mono',monospace",color:"#94a3b8",letterSpacing:1 }}>QUERY LOG</h3>
                    <button style={{ background:"rgba(16,185,129,0.1)",border:"1px solid rgba(16,185,129,0.2)",
                      color:"#10b981",padding:"6px 14px",borderRadius:8,cursor:"pointer",
                      fontSize:12,fontFamily:"'JetBrains Mono',monospace" }}
                      onClick={()=>{
                        const data = JSON.stringify({ exportDate:new Date().toISOString(), platform:"MedGuard QHI v0.1.0",
                          metrics:sessionMetrics, queries:sessionHistory.map(h=>({
                            question:h.question, response:h.content.substring(0,200), qhi:h.qhiResult.qhi,
                            gate:h.qhiResult.gate, confidence:h.qhiResult.confidence, specialty:h.qhiResult.specialty,
                            entities:h.qhiResult.entities, uncertainty:h.qhiResult.uncertainty,
                            riskScore:h.qhiResult.riskScore, violation:h.qhiResult.violation,
                            dangerMatches:h.qhiResult.dangerMatches, timestamp:h.qhiResult.timestamp,
                          })) },null,2);
                        const b=new Blob([data],{type:"application/json"});
                        const u=URL.createObjectURL(b); const a=document.createElement("a");
                        a.href=u; a.download=`qhi_research_${Date.now()}.json`; a.click(); URL.revokeObjectURL(u);
                      }}>Export JSON ↓</button>
                  </div>
                  {sessionHistory.map((h,i) => (
                    <div key={i} style={{ display:"flex",alignItems:"center",gap:12,padding:"10px 0",
                      borderBottom:i<sessionHistory.length-1?"1px solid #1e293b":"none" }}>
                      <span style={{ fontSize:11,color:"#475569",fontFamily:"'JetBrains Mono',monospace",width:24 }}>{i+1}</span>
                      <GateBadge gate={h.qhiResult.gate} />
                      <span style={{ fontSize:12,fontFamily:"'JetBrains Mono',monospace",color:"#e2e8f0",fontWeight:600,width:50 }}>
                        {h.qhiResult.qhi.toFixed(2)}
                      </span>
                      <span style={{ flex:1,color:"#94a3b8",fontSize:13,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap" }}>
                        {h.question}
                      </span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {/* ═══ ABOUT ═══ */}
        {view === "about" && (
          <div style={{ maxWidth:800,margin:"0 auto",padding:"40px 24px" }}>
            <h2 style={{ fontSize:32,fontWeight:700,marginBottom:24,fontFamily:"'Instrument Serif',Georgia,serif",color:"#f8fafc" }}>
              About MedGuard QHI
            </h2>
            {[
              { title:"The Problem", content:"When a clinical AI hallucinates, it outputs dangerous misinformation in the same confident tone as correct answers. Existing detection methods are binary (hallucinated or not), require a second GPU-powered LLM, and produce no regulatory output." },
              { title:"Our Solution", content:"QHI-Probe trains three tiny classifiers — Uncertainty (Probe-C), Risk (Probe-R), and Violation (Probe-V) — that produce a single auditable hallucination severity score from 0–25 in under 0.1ms on CPU. The multiplicative formula QHI = U × R × V × 5 ensures all three signals must agree, preventing false alarms." },
              { title:"Who Is This For", content:"Doctors & clinicians verifying AI-assisted decisions. Hospitals deploying AI with ISO 14971 compliance. The general public seeking trustworthy medical information. Researchers benchmarking LLMs for clinical hallucination." },
            ].map((s,i) => (
              <div key={i} className="card" style={{ padding:32,marginBottom:24 }}>
                <h3 style={{ fontSize:18,fontWeight:700,color:"#10b981",marginBottom:12,fontFamily:"'JetBrains Mono',monospace" }}>{s.title}</h3>
                <p style={{ color:"#94a3b8",lineHeight:1.8,fontSize:15 }}>{s.content}</p>
              </div>
            ))}
            <div className="card" style={{ padding:32,marginBottom:24 }}>
              <h3 style={{ fontSize:18,fontWeight:700,color:"#10b981",marginBottom:12,fontFamily:"'JetBrains Mono',monospace" }}>Open Source</h3>
              <p style={{ color:"#94a3b8",lineHeight:1.8,fontSize:15,marginBottom:16 }}>
                MIT License — free for research and commercial use.
              </p>
              <div style={{ padding:16,background:"#020617",borderRadius:12,border:"1px solid #1e293b",
                fontFamily:"'JetBrains Mono',monospace",fontSize:13,color:"#10b981" }}>
                <div style={{ color:"#64748b" }}># Clone and run</div>
                <div>git clone https://github.com/Roxrite0509/QHI.git</div>
                <div>cd QHI/web && npm install && npm run dev</div>
              </div>
            </div>
            <div className="card" style={{ padding:32,borderColor:"#10b98133" }}>
              <h3 style={{ fontSize:18,fontWeight:700,color:"#10b981",marginBottom:12,fontFamily:"'JetBrains Mono',monospace" }}>Citation</h3>
              <div style={{ padding:16,background:"#020617",borderRadius:12,border:"1px solid #1e293b",
                fontFamily:"'JetBrains Mono',monospace",fontSize:12,color:"#94a3b8",lineHeight:1.8 }}>
                @misc&#123;pranav2025qhiprobe,<br/>
                &nbsp;&nbsp;title = &#123;QHI-Probe: Quantified Hallucination Index for Clinical LLMs&#125;,<br/>
                &nbsp;&nbsp;author = &#123;Pranav&#125;, year = &#123;2025&#125;,<br/>
                &nbsp;&nbsp;url = &#123;https://github.com/Roxrite0509/QHI&#125;<br/>
                &#125;
              </div>
            </div>
          </div>
        )}
      </div>

      <footer style={{ borderTop:"1px solid #1e293b",padding:24,textAlign:"center",fontSize:12,
        color:"#475569",fontFamily:"'JetBrains Mono',monospace",marginTop:view==="chat"?0:60 }}>
        <div>MedGuard QHI v0.1.0 · Open Source · MIT License · Not medical advice</div>
        <div style={{ marginTop:4 }}>QHI-Probe: Quantified Hallucination Index for Clinical LLMs © 2025</div>
      </footer>
    </div>
  );
}
