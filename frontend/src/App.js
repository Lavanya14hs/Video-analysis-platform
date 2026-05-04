import React, { useState, useRef, useCallback, } from "react";
import "./App.css";


// ─── LIGHTBOX ────────────────────────────────────────────
const lightboxStyles = `
  .lightbox-overlay {
    position: fixed;
    inset: 0;
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(4, 8, 18, 0);
    backdrop-filter: blur(0px);
    transition: background 0.25s ease, backdrop-filter 0.25s ease;
    cursor: zoom-out;
  }
  .lightbox-overlay.open {
    background: rgba(4, 8, 18, 0.92);
    backdrop-filter: blur(18px);
  }
  .lightbox-img-wrap {
    position: relative;
    transform: scale(0.6);
    opacity: 0;
    transition: transform 0.28s cubic-bezier(0.34, 1.56, 0.64, 1), opacity 0.22s ease;
    max-width: 90vw;
    max-height: 88vh;
    cursor: default;
  }
  .lightbox-overlay.open .lightbox-img-wrap {
    transform: scale(1);
    opacity: 1;
  }
  .lightbox-img {
    display: block;
    max-width: 90vw;
    max-height: 82vh;
    width: auto;
    height: auto;
    border-radius: 12px;
    box-shadow: 0 32px 80px rgba(0,0,0,0.8), 0 0 0 1px rgba(255,255,255,0.07);
    object-fit: contain;
  }
  .lightbox-close {
    position: absolute;
    top: -14px;
    right: -14px;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: var(--bg-raised);
    border: 1px solid var(--border-hi);
    color: var(--text-2);
    font-size: 14px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.15s, color 0.15s, transform 0.15s;
    line-height: 1;
  }
  .lightbox-close:hover {
    background: var(--accent-red);
    color: #fff;
    transform: scale(1.12);
    border-color: transparent;
  }
  .lightbox-meta {
    position: absolute;
    bottom: -36px;
    left: 0; right: 0;
    text-align: center;
    font-size: 12px;
    color: var(--text-3);
    font-family: var(--font-mono);
    letter-spacing: 0.3px;
  }
  .event-thumb {
    cursor: zoom-in !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease !important;
  }
  .event-thumb:hover {
    transform: scale(1.06) !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.5) !important;
    border-color: var(--accent-blue) !important;
  }
  .event-thumb-placeholder {
    cursor: zoom-in !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease !important;
  }
  .event-thumb-placeholder:hover {
    transform: scale(1.06) !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.5) !important;
  }
`;

function LightboxStyle() {
  return React.createElement('style', null, lightboxStyles);
}

function Lightbox({ src, label, onClose }) {
  const [open, setOpen] = React.useState(false);

  React.useEffect(() => {
    const t = setTimeout(() => setOpen(true), 10);
    return () => clearTimeout(t);
  }, []);

  React.useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className={`lightbox-overlay ${open ? "open" : ""}`}
      onClick={onClose}
    >
      <div className="lightbox-img-wrap" onClick={e => e.stopPropagation()}>
        <img className="lightbox-img" src={src} alt={label || "snapshot"} />
        <button className="lightbox-close" onClick={onClose}>✕</button>
        {label && <div className="lightbox-meta">{label}</div>}
      </div>
    </div>
  );
}

// ─── UTILS ───────────────────────────────────────────────
function formatBytes(bytes) {
  if (!bytes) return "";
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatElapsed(s) {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

// ─── TOP BAR ─────────────────────────────────────────────
function TopBar() {
  return (
    <header className="topbar">
      <div className="topbar-right">
        <div className="topbar-status">
          <div className="dot-live" />
          System online
        </div>
      </div>
    </header>
  );
}

// ─── UPLOAD PAGE ──────────────────────────────────────────
function UploadPage({ file, setFile, onAnalyze, loading, progress, elapsed }) {
  const inputRef = useRef();
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("video/")) setFile(f);
  }, [setFile]);

  const handleDrag = (e) => {
    e.preventDefault();
    setDragging(e.type === "dragover");
  };

  const stages = [
    { label: "Upload",   threshold: 0 },
    { label: "Decode",   threshold: 15 },
    { label: "Detect",   threshold: 30 },
    { label: "Classify", threshold: 65 },
    { label: "Report",   threshold: 90 },
  ];

  return (
    <div className="page-content">
      <div className="page-header">
        <h1 className="page-title">Video Analysis Platform For Incident Detection and Response</h1>
        <p className="page-subtitle">Upload a video to automatically detect fights, accidents and safety events using AI</p>
      </div>

      {/* Drop zone */}
      <div
        className={`upload-zone ${dragging ? "drag-active" : ""}`}
        onClick={() => inputRef.current.click()}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
      >
        <div className="upload-icon">📹</div>
        <div className="upload-title">Drop your video here</div>
        <div className="upload-hint">or click to browse — MP4, MOV, AVI, MKV supported</div>
        <div className="file-types">
          {["MP4", "MOV", "AVI", "MKV", "WEBM"].map(t => (
            <span key={t} className="file-type-pill">{t}</span>
          ))}
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          onChange={e => setFile(e.target.files[0] || null)}
        />
      </div>

      {/* Selected file */}
      {file && !loading && (
        <div className="file-selected">
          <div className="file-info">
            <span className="file-icon">🎞️</span>
            <div>
              <div className="file-meta-name">{file.name}</div>
              <div className="file-meta-size">{formatBytes(file.size)}</div>
            </div>
          </div>
          <button className="file-remove" onClick={e => { e.stopPropagation(); setFile(null); }}>✕</button>
        </div>
      )}

      {/* Analyze button */}
      {!loading && (
        <div className="analyze-row">
          <button
            className="btn btn-primary"
            disabled={!file}
            onClick={onAnalyze}
          >
            <span className="btn-icon">🚀</span>
            Analyze Video
          </button>
          {!file && <span className="analyze-note">Select a video to continue</span>}
        </div>
      )}

      {/* Processing card */}
      {loading && (
        <div className="processing-card">
          <div className="processing-header">
            <div className="processing-title">
              <div className="spinner" />
              Analyzing video...
            </div>
            <div className="processing-timer">⏱ {formatElapsed(elapsed)}</div>
          </div>

          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <div className="progress-labels">
            <span>{progress}% complete</span>
            <span>{file?.name}</span>
          </div>

          <div className="processing-stages">
            {stages.map((s, i) => {
              const isDone   = progress > s.threshold + 15;
              const isActive = !isDone && progress >= s.threshold;
              return (
                <div
                  key={i}
                  className={`stage-pill ${isActive ? "active" : ""} ${isDone ? "done" : ""}`}
                >
                  <div className="stage-dot" />
                  {s.label}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── CROWD LEVEL COLOR ───────────────────────────────────
function crowdMeta(level) {
  switch (level) {
    case "Busy":         return { color: "#f59e0b", bg: "rgba(245,158,11,0.10)", border: "rgba(245,158,11,0.25)", icon: "👥" };
    case "Crowded":      return { color: "#f97316", bg: "rgba(249,115,22,0.10)", border: "rgba(249,115,22,0.25)", icon: "👥" };
    case "Very Crowded": return { color: "#ef4444", bg: "rgba(239,68,68,0.10)",  border: "rgba(239,68,68,0.25)",  icon: "🚨" };
    case "Critical Crowd": return { color: "#dc2626", bg: "rgba(220,38,38,0.15)", border: "rgba(220,38,38,0.35)", icon: "🆘" };
    default:             return { color: "#8a94a8", bg: "rgba(138,148,168,0.08)", border: "rgba(138,148,168,0.2)", icon: "👥" };
  }
}

// ─── RESULT PAGES ─────────────────────────────────────────
function ResultSection({ result, elapsed, onReset }) {
  const [tab, setTab] = useState("overview");
  const [lightbox, setLightbox] = useState(null); // { src, label }
  const openLightbox = useCallback((src, label) => setLightbox({ src, label }), []);
  const closeLightbox = useCallback(() => setLightbox(null), []);

  const fightCount    = result.alerts.filter(a => a.fight).length;
  const accidentCount = result.alerts.filter(a => a.accident).length;
  const crowdCount    = result.alerts.filter(a => a.crowd).length;
  const totalEvents   = result.alerts.length;

  const summary = result.alerts.reduce((acc, a) => {
    Object.entries(a.counts || {}).forEach(([k, v]) => {
      acc[k] = Math.max(acc[k] || 0, v);
    });
    return acc;
  }, {});

  const hasAlerts = totalEvents > 0;

  const fightEvents    = result.alerts.filter(a => a.fight);
  const accidentEvents = result.alerts.filter(a => a.accident);
  const crowdEvents    = result.alerts.filter(a => a.crowd);

  return (
    <div className="app-shell">
      <TopBar />

      {/* Nav */}
      <nav className="nav-tabs">
        {[
          { id: "overview", label: "Overview", icon: "📊", badge: null },
          { id: "events",   label: "Incidents", icon: "🚨", badge: (fightCount + accidentCount) || null },
          { id: "crowd",    label: "Crowd",    icon: "👥", badge: crowdCount || null },
          { id: "report",   label: "Report",   icon: "📄", badge: null },
        ].map(t => (
          <button
            key={t.id}
            className={`nav-tab ${tab === t.id ? "active" : ""}`}
            onClick={() => setTab(t.id)}
          >
            {t.icon} {t.label}
            {t.badge != null && (
              <span className="nav-tab-badge">{t.badge}</span>
            )}
          </button>
        ))}
      </nav>

      <div className="page-content">

        {/* ── OVERVIEW ── */}
        {tab === "overview" && (
          <>
            <div className="page-header">
              <h1 className="page-title">Analysis Complete</h1>
              <p className="page-subtitle">
                Processed in {formatElapsed(elapsed)} · {totalEvents} event{totalEvents !== 1 ? "s" : ""} recorded
              </p>
            </div>

            {hasAlerts && (
              <div className="alert-banner">
                <span className="alert-banner-icon">⚠️</span>
                <div className="alert-banner-text">
                  <div className="alert-banner-title">Incidents Detected</div>
                  <div className="alert-banner-sub">
                    {fightCount} fight{fightCount !== 1 ? "s" : ""} · {accidentCount} accident{accidentCount !== 1 ? "s" : ""} · {crowdCount} crowd alert{crowdCount !== 1 ? "s" : ""}
                  </div>
                </div>
                <button className="btn btn-danger" onClick={() => setTab("events")}>View Events →</button>
              </div>
            )}

            <div className="stats-grid">
              <div className="stat-card" style={{ "--card-accent": "linear-gradient(90deg,#3d7fff,#00c8b4)" }}>
                <div className="stat-label">Total Events</div>
                <div className="stat-value" style={{ color: "var(--accent-blue)" }}>{totalEvents}</div>
                <div className="stat-sub">All detected incidents</div>
              </div>
              <div className="stat-card" style={{ "--card-accent": "linear-gradient(90deg,#f05252,#f59e0b)" }}>
                <div className="stat-label">Fights</div>
                <div className="stat-value" style={{ color: "var(--accent-red)" }}>{fightCount}</div>
                <div className="stat-sub">Confirmed violence</div>
              </div>
              <div className="stat-card" style={{ "--card-accent": "linear-gradient(90deg,#f59e0b,#f05252)" }}>
                <div className="stat-label">Accidents</div>
                <div className="stat-value" style={{ color: "var(--accent-amber)" }}>{accidentCount}</div>
                <div className="stat-sub">Vehicle incidents</div>
              </div>
              <div className="stat-card" style={{ "--card-accent": "linear-gradient(90deg,#a855f7,#3d7fff)" }}>
                <div className="stat-label">Crowd Alerts</div>
                <div className="stat-value" style={{ color: "#a855f7" }}>{crowdCount}</div>
                <div className="stat-sub">Density events</div>
              </div>
              <div className="stat-card" style={{ "--card-accent": "linear-gradient(90deg,#22c55e,#00c8b4)" }}>
                <div className="stat-label">Analysis Time</div>
                <div className="stat-value" style={{ fontSize: "22px", color: "var(--accent-green)", paddingTop: "4px" }}>{formatElapsed(elapsed)}</div>
                <div className="stat-sub">Processing duration</div>
              </div>
            </div>

            {Object.keys(summary).length > 0 && (
              <>
                <div className="section-header">
                  <div className="section-title">Objects Detected</div>
                  <span className="section-count">{Object.keys(summary).length} classes</span>
                </div>
                <div className="objects-grid">
                  {Object.entries(summary).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                    <div key={k} className="obj-card">
                      <span className="obj-name">{k}</span>
                      <span className="obj-count">{v}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </>
        )}

        {/* ── INCIDENTS ── */}
        {tab === "events" && (
          <>
            <div className="page-header">
              <h1 className="page-title">Incident Timeline</h1>
              <p className="page-subtitle">Fights and accidents — confirmed after multi-frame validation</p>
            </div>

            {fightEvents.length === 0 && accidentEvents.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">✅</div>
                <div className="empty-title">No fights or accidents detected</div>
                <div className="empty-hint">Multi-frame validation found no confirmed incidents.</div>
              </div>
            ) : (
              <div className="timeline">
                {result.alerts.filter(a => a.fight || a.accident).map((a, i) => {
                  const imgUrl = a.snapshot || null;
                  const confPct = a.fight_conf ? Math.round(a.fight_conf * 100) : null;
                  return (
                    <div className="event-card" key={i}>
                      {imgUrl
                        ? <img className="event-thumb" src={imgUrl} alt="snapshot" onClick={() => openLightbox(imgUrl, a.time_str)} />
                        : <div className="event-thumb-placeholder" onClick={() => openLightbox(null, a.time_str)}>{a.fight ? "🔥" : "🚗"}</div>
                      }
                      <div className="event-body">
                        <div className="event-time">
                          ⏱ {a.time_str}
                          {a.flow_mag != null && (
                            <span style={{ marginLeft: 10, fontSize: 11, color: "var(--text-3)" }}>
                              motion: {a.flow_mag}
                            </span>
                          )}
                        </div>
                        <div className="event-badges">
                          {a.fight && (
                            <span className="badge badge-fight">
                              🔥 Fight Detected
                              {confPct && <span style={{ marginLeft: 6, opacity: 0.75, fontSize: 11 }}>{confPct}% conf</span>}
                            </span>
                          )}
                          {a.accident && (
                            <span className="badge badge-accident">
                              🚗 Accident · {a.accident_type}
                            </span>
                          )}
                        </div>
                        {a.counts && (
                          <div className="event-objects">
                            {Object.entries(a.counts).map(([k, v]) => (
                              <span className="object-chip" key={k}>{k} ×{v}</span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </>
        )}

        {/* ── CROWD ── */}
        {tab === "crowd" && (
          <>
            <div className="page-header">
              <h1 className="page-title">Crowd Monitoring</h1>
              <p className="page-subtitle">Person density alerts — Busy / Crowded / Very Crowded / Critical</p>
            </div>

            {crowdEvents.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">🟢</div>
                <div className="empty-title">No crowd alerts</div>
                <div className="empty-hint">Person density stayed below the threshold throughout the video.</div>
              </div>
            ) : (
              <div className="timeline">
                {crowdEvents.map((a, i) => {
                  const imgUrl = a.snapshot || null;
                  const meta   = crowdMeta(a.crowd_level);
                  const personCount = a.counts?.person || 0;
                  return (
                    <div className="event-card" key={i}>
                      {imgUrl
                        ? <img className="event-thumb" src={imgUrl} alt="snapshot" onClick={() => openLightbox(imgUrl, a.time_str)} />
                        : <div className="event-thumb-placeholder" onClick={() => openLightbox(null, a.time_str)}>{meta.icon}</div>
                      }
                      <div className="event-body">
                        <div className="event-time">⏱ {a.time_str}</div>
                        <div className="event-badges">
                          <span className="badge" style={{ background: meta.bg, color: meta.color, border: `1px solid ${meta.border}` }}>
                            {meta.icon} {a.crowd_level}
                          </span>
                          <span className="badge" style={{ background: "var(--bg-raised)", color: "var(--text-2)", border: "1px solid var(--border)" }}>
                            👤 {personCount} persons
                          </span>
                          {a.density_score != null && (
                            <span className="badge" style={{ background: "var(--bg-raised)", color: "var(--text-2)", border: "1px solid var(--border)" }}>
                              density {Math.round(a.density_score * 100)}%
                            </span>
                          )}
                        </div>
                        {a.dense_zones && a.dense_zones.length > 0 && (
                          <div className="event-objects" style={{ marginTop: 6 }}>
                            {a.dense_zones.map((z, zi) => (
                              <span className="object-chip" key={zi}>
                                zone {zi + 1}: {z.count} persons
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </>
        )}

        {/* ── REPORT ── */}
        {tab === "report" && (
          <>
            <div className="page-header">
              <h1 className="page-title">Incident Report</h1>
              <p className="page-subtitle">Machine-generated summary of all detected events</p>
            </div>
            <div className="report-box">{result.report || "No report generated."}</div>
          </>
        )}
      </div>

      <div className="action-bar">
        <span className="action-bar-info">
          Analysis complete · {fightCount} fight{fightCount !== 1 ? "s" : ""} · {accidentCount} accident{accidentCount !== 1 ? "s" : ""} · {crowdCount} crowd alert{crowdCount !== 1 ? "s" : ""}
        </span>
        <button className="btn btn-secondary" onClick={onReset}>← Analyze New Video</button>
      </div>

      <LightboxStyle />
      {lightbox && (
        <Lightbox src={lightbox.src} label={lightbox.label} onClose={closeLightbox} />
      )}
    </div>
  );
}

// ─── ROOT APP ─────────────────────────────────────────────
export default function App() {
  const [file,     setFile]     = useState(null);
  const [progress, setProgress] = useState(0);
  const [result,   setResult]   = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [elapsed,  setElapsed]  = useState(0);

  const timerRef = useRef(null);

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setProgress(0);
    setElapsed(0);
    setResult(null);

    timerRef.current = setInterval(() => setElapsed(p => p + 1), 1000);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("http://127.0.0.1:8000/analyze-stream", {
        method: "POST",
        body: formData,
      });

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value);
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const json = JSON.parse(line.replace("data: ", ""));
            if (json.progress !== undefined) setProgress(json.progress);
            if (json.done) {
              const safe = json.result || {};
              setResult({
                alerts: safe.alerts || [],
                report: safe.report || "",
              });
              clearInterval(timerRef.current);
              setLoading(false);
            }
          } catch (_) {}
        }
      }
    } catch (err) {
      console.error("Analyze error:", err);
      clearInterval(timerRef.current);
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setProgress(0);
    setElapsed(0);
    setLoading(false);
    clearInterval(timerRef.current);
  };

  if (result) {
    return (
      <ResultSection
        result={result}
        elapsed={elapsed}
        onReset={handleReset}
      />
    );
  }

  return (
    <div className="app-shell">
      <TopBar />
      <UploadPage
        file={file}
        setFile={setFile}
        onAnalyze={handleAnalyze}
        loading={loading}
        progress={progress}
        elapsed={elapsed}
      />
    </div>
  );
}