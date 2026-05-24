import React, { useState, useRef, useCallback } from "react";
import "./App.css";

// ─── LIGHTBOX ────────────────────────────────────────────
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
    <div className={`lightbox-overlay ${open ? "open" : ""}`} onClick={onClose}>
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

// ─── BACKGROUND SVG ──────────────────────────────────────
function BackgroundArt() {
  return (
    <svg
      style={{
        position: "fixed",
        inset: 0,
        width: "100%",
        height: "100%",
        zIndex: 0,
        pointerEvents: "none",
        opacity: 0.5,
      }}
      viewBox="0 0 1400 900"
      preserveAspectRatio="xMidYMid slice"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <radialGradient id="rg1" cx="20%" cy="10%" r="55%">
          <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.12" />
          <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
        </radialGradient>
        <radialGradient id="rg2" cx="80%" cy="90%" r="50%">
          <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.09" />
          <stop offset="100%" stopColor="#06b6d4" stopOpacity="0" />
        </radialGradient>
        <radialGradient id="rg3" cx="60%" cy="30%" r="35%">
          <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.06" />
          <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0" />
        </radialGradient>
        <filter id="blur1">
          <feGaussianBlur stdDeviation="60" />
        </filter>
      </defs>
      <rect width="1400" height="900" fill="url(#rg1)" />
      <rect width="1400" height="900" fill="url(#rg2)" />
      <rect width="1400" height="900" fill="url(#rg3)" />

      {/* Corner bracket — top left */}
      <path d="M0 120 L0 0 L140 0" stroke="rgba(59,130,246,0.2)" strokeWidth="1.5" fill="none" />
      {/* Corner bracket — top right */}
      <path d="M1260 0 L1400 0 L1400 120" stroke="rgba(6,182,212,0.15)" strokeWidth="1.5" fill="none" />
      {/* Corner bracket — bottom left */}
      <path d="M0 780 L0 900 L140 900" stroke="rgba(59,130,246,0.1)" strokeWidth="1.5" fill="none" />
      {/* Corner bracket — bottom right */}
      <path d="M1260 900 L1400 900 L1400 780" stroke="rgba(6,182,212,0.1)" strokeWidth="1.5" fill="none" />

      {/* Horizontal scan line */}
      <line x1="0" y1="460" x2="1400" y2="460" stroke="rgba(59,130,246,0.04)" strokeWidth="1" />

      {/* Decorative hexagon ring */}
      <polygon
        points="700,60 760,95 760,165 700,200 640,165 640,95"
        stroke="rgba(59,130,246,0.08)"
        strokeWidth="1"
        fill="none"
      />
      <polygon
        points="700,80 744,105 744,155 700,180 656,155 656,105"
        stroke="rgba(59,130,246,0.05)"
        strokeWidth="1"
        fill="none"
      />

      {/* Crosshair marks */}
      <g stroke="rgba(6,182,212,0.15)" strokeWidth="1">
        <line x1="84" y1="80" x2="96" y2="80" />
        <line x1="90" y1="74" x2="90" y2="86" />
        <line x1="1304" y1="80" x2="1316" y2="80" />
        <line x1="1310" y1="74" x2="1310" y2="86" />
        <line x1="84" y1="820" x2="96" y2="820" />
        <line x1="90" y1="814" x2="90" y2="826" />
      </g>

      {/* Subtle arc curves */}
      <path d="M 0 700 Q 350 500 700 600 T 1400 400" stroke="rgba(59,130,246,0.04)" strokeWidth="1" fill="none" />
      <path d="M 0 500 Q 350 300 700 400 T 1400 200" stroke="rgba(6,182,212,0.03)" strokeWidth="1" fill="none" />
    </svg>
  );
}

// ─── ANIMATED DOTS ────────────────────────────────────────
function FloatingDots() {
  const dots = [
    { cx: "8%",  cy: "20%", r: 1.5, delay: "0s",   dur: "4s"  },
    { cx: "92%", cy: "35%", r: 1,   delay: "1s",   dur: "5s"  },
    { cx: "15%", cy: "75%", r: 1,   delay: "2s",   dur: "6s"  },
    { cx: "75%", cy: "80%", r: 1.5, delay: "0.5s", dur: "4.5s"},
    { cx: "50%", cy: "10%", r: 1,   delay: "1.5s", dur: "5.5s"},
    { cx: "30%", cy: "55%", r: 0.8, delay: "3s",   dur: "7s"  },
  ];
  return (
    <svg
      style={{ position: "fixed", inset: 0, width: "100%", height: "100%", zIndex: 0, pointerEvents: "none" }}
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
    >
      {dots.map((d, i) => (
        <circle key={i} cx={d.cx} cy={d.cy} r={d.r} fill="rgba(59,130,246,0.45)">
          <animate attributeName="opacity" values="0.2;0.9;0.2" dur={d.dur} begin={d.delay} repeatCount="indefinite" />
          <animate attributeName="r" values={`${d.r};${d.r * 1.6};${d.r}`} dur={d.dur} begin={d.delay} repeatCount="indefinite" />
        </circle>
      ))}
    </svg>
  );
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

// ─── HERO SECTION ─────────────────────────────────────────
function HeroSection() {
  return (
    <div className="hero-wrapper">
      <div className="hero-eyebrow">
        AI-Powered Surveillance Intelligence
      </div>

      <h1 className="hero-title">
        Incident Detection &amp;<br />
        <span className="gradient-word">Response Platform</span>
      </h1>

      <p className="hero-desc">
        Enterprise-grade real-time video analysis powered by multi-model AI pipelines.
        Automatically detect fights, accidents, fire/smoke hazards, and crowd anomalies from any video source —
        with frame-level accuracy, optical-flow validation, and temporal consensus verification.
      </p>

      <div className="hero-pills">
        <span className="hero-pill blue"><span className="pill-dot" />YOLOv8 Object Detection</span>
        <span className="hero-pill cyan"><span className="pill-dot" />Optical Flow Analysis</span>
        <span className="hero-pill red"><span className="pill-dot" />Fight &amp; Violence Detection</span>
        <span className="hero-pill magenta"><span className="pill-dot" />Fire &amp; Smoke Alerts</span>
        <span className="hero-pill green"><span className="pill-dot" />Crowd Density Monitoring</span>
      </div>
      <div className="hero-metrics">
        <div className="metric-card">
          <div className="metric-label">Accuracy</div>
          <div className="metric-value blue">94.7%</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Avg Process Time</div>
          <div className="metric-value green">&lt; 4min</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Detection Classes</div>
          <div className="metric-value amber">80+</div>
        </div>
      </div>
    </div>
  );
}

// ─── CROWD LEVEL META ──────────────────────────────────────
function crowdMeta(level) {
  switch (level) {
    case "Busy":           return { color: "#f59e0b", bg: "rgba(245,158,11,0.10)", border: "rgba(245,158,11,0.25)", icon: "👥" };
    case "Crowded":        return { color: "#f97316", bg: "rgba(249,115,22,0.10)", border: "rgba(249,115,22,0.25)", icon: "👥" };
    case "Very Crowded":   return { color: "#ef4444", bg: "rgba(239,68,68,0.10)",  border: "rgba(239,68,68,0.25)",  icon: "🚨" };
    case "Critical Crowd": return { color: "#dc2626", bg: "rgba(220,38,38,0.15)",  border: "rgba(220,38,38,0.35)", icon: "🆘" };
    default:               return { color: "#7a8799", bg: "rgba(122,135,153,0.08)", border: "rgba(122,135,153,0.2)", icon: "👥" };
  }
}

// ─── RESULT SECTION ───────────────────────────────────────
function ResultSection({ result, elapsed, onReset }) {
  const [tab, setTab] = useState("overview");
  const [lightbox, setLightbox] = useState(null);
  const openLightbox = useCallback((src, label) => setLightbox({ src, label }), []);
  const closeLightbox = useCallback(() => setLightbox(null), []);

  const fightCount    = result.alerts.filter(a => a.fight).length;
  const accidentCount = result.alerts.filter(a => a.accident).length;
  const crowdCount    = result.alerts.filter(a => a.crowd).length;
  const fireCount     = result.alerts.reduce((acc, a) => acc + (a.fire_events?.includes("Fire") ? 1 : 0), 0);
  const smokeCount    = result.alerts.reduce((acc, a) => acc + (a.fire_events?.includes("Smoke") ? 1 : 0), 0);
  const totalEvents   = result.alerts.length;

  const summary = result.object_counts || result.alerts.reduce((acc, a) => {
    Object.entries(a.counts || {}).forEach(([k, v]) => {
      acc[k] = Math.max(acc[k] || 0, v);
    });
    return acc;
  }, {});

  const hasAlerts      = totalEvents > 0;
  const crowdEvents    = result.alerts.filter(a => a.crowd);
  const incidentEvents = result.alerts.filter(a => a.fight || a.accident || (a.fire_events?.length || 0) > 0);

  return (
    <div className="app-shell result-shell">
      <BackgroundArt />
      <FloatingDots />
      <TopBar />

      <nav className="nav-tabs">
        {[
          { id: "overview", label: "Overview",  icon: "📊", badge: null },
          { id: "events",   label: "Incidents", icon: "🚨", badge: incidentEvents.length || null },
          { id: "crowd",    label: "Crowd",     icon: "👥", badge: crowdCount || null },
          { id: "report",   label: "Report",    icon: "📄", badge: null },
        ].map(t => (
          <button
            key={t.id}
            className={`nav-tab ${tab === t.id ? "active" : ""}`}
            onClick={() => setTab(t.id)}
          >
            {t.icon}&nbsp;{t.label}
            {t.badge != null && <span className="nav-tab-badge">{t.badge}</span>}
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
                  {(fireCount + smokeCount) > 0 && ` · ${fireCount + smokeCount} fire/smoke`}
                  </div>
                </div>
                <button className="btn btn-danger" onClick={() => setTab("events")}>View Events →</button>
              </div>
            )}

            <div className="stats-grid">
              <div className="stat-card" style={{ "--card-accent": "linear-gradient(90deg,#3b82f6,#06b6d4)" }}>
                <div className="stat-label">Total Events</div>
                <div className="stat-value" style={{ color: "var(--accent-blue)" }}>{totalEvents}</div>
                <div className="stat-sub">All detected incidents</div>
              </div>
              <div className="stat-card" style={{ "--card-accent": "linear-gradient(90deg,#ef4444,#f59e0b)" }}>
                <div className="stat-label">Fights</div>
                <div className="stat-value" style={{ color: "var(--accent-red)" }}>{fightCount}</div>
                <div className="stat-sub">Confirmed violence</div>
              </div>
              <div className="stat-card" style={{ "--card-accent": "linear-gradient(90deg,#f59e0b,#ef4444)" }}>
                <div className="stat-label">Accidents</div>
                <div className="stat-value" style={{ color: "var(--accent-amber)" }}>{accidentCount}</div>
                <div className="stat-sub">Vehicle incidents</div>
              </div>
              <div className="stat-card" style={{ "--card-accent": "linear-gradient(90deg,#8b5cf6,#3b82f6)" }}>
                <div className="stat-label">Crowd Alerts</div>
                <div className="stat-value" style={{ color: "#8b5cf6" }}>{crowdCount}</div>
                <div className="stat-sub">Density events</div>
              </div>
              <div className="stat-card" style={{ "--card-accent": "linear-gradient(90deg,#ec4899,#8b5cf6)" }}>
                <div className="stat-label">Fire / Smoke Alerts</div>
                <div className="stat-value" style={{ color: "#ec4899" }}>{fireCount + smokeCount}</div>
                <div className="stat-sub">Flame and smoke events</div>
              </div>
              <div className="stat-card" style={{ "--card-accent": "linear-gradient(90deg,#10b981,#06b6d4)" }}>
                <div className="stat-label">Analysis Time</div>
                <div className="stat-value" style={{ fontSize: "24px", color: "var(--accent-green)", paddingTop: "6px" }}>{formatElapsed(elapsed)}</div>
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
              <p className="page-subtitle">Confirmed incidents — fights, accidents and fire/smoke alerts</p>
            </div>

            {incidentEvents.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">✅</div>
                <div className="empty-title">No confirmed incidents detected</div>
                <div className="empty-hint">Multi-frame validation found no confirmed events.</div>
              </div>
            ) : (
              <div className="timeline">
                {incidentEvents.map((a, i) => {
                  const imgUrl  = a.snapshot || null;
                  const confPct = a.fight_conf ? Math.round(a.fight_conf * 100) : null;
                  const eventLabel = [
                    a.fight && 'Fight',
                    a.accident && a.accident_type && `Accident (${a.accident_type})`,
                    a.fire_events?.length > 0 && a.fire_events.join(', '),
                  ].filter(Boolean).join(' · ');
                  const lightboxLabel = eventLabel ? `${a.time_str} · ${eventLabel}` : a.time_str;
                  const placeholder = a.fire_events?.includes('Fire')
                    ? '🔥'
                    : a.fire_events?.includes('Smoke')
                      ? '💨'
                      : a.fight
                        ? '🔥'
                        : a.accident
                          ? '🚗'
                          : '🚨';
                  return (
                    <div className="event-card" key={a.time_str || i}>
                      {imgUrl
                        ? <img className="event-thumb" src={imgUrl} alt="snapshot" onClick={() => openLightbox(imgUrl, lightboxLabel)} />
                        : <div className="event-thumb-placeholder" onClick={() => openLightbox(null, lightboxLabel)}>{placeholder}</div>
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
                              {confPct && <span style={{ marginLeft: 6, opacity: 0.75, fontSize: 10 }}>{confPct}% conf</span>}
                            </span>
                          )}
                          {a.accident && (
                            <span className="badge badge-accident">
                              🚗 Accident · {a.accident_type}
                            </span>
                          )}
                          {a.fire_events?.includes("Fire") && (
                            <span className="badge badge-fire">
                              🔥 Fire
                            </span>
                          )}
                          {a.fire_events?.includes("Smoke") && (
                            <span className="badge badge-smoke">
                              💨 Smoke
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
                  const imgUrl      = a.snapshot || null;
                  const meta        = crowdMeta(a.crowd_level);
                  const personCount = a.counts?.person || 0;
                  return (
                    <div className="event-card" key={a.time_str || i}>
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
                              <span className="object-chip" key={zi}>zone {zi + 1}: {z.count} persons</span>
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
          {(fireCount + smokeCount) > 0 && ` · ${fireCount + smokeCount} fire/smoke`}
        </span>
        <button className="btn btn-secondary" onClick={onReset}>← Analyze New Video</button>
      </div>

      {lightbox && <Lightbox src={lightbox.src} label={lightbox.label} onClose={closeLightbox} />}
    </div>
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
    { label: "Upload",   threshold: 0  },
    { label: "Decode",   threshold: 15 },
    { label: "Detect",   threshold: 30 },
    { label: "Classify", threshold: 65 },
    { label: "Report",   threshold: 90 },
  ];

  return (
    <>
      {/* Hero section */}
      <HeroSection />
      <div className="hero-divider" />

      {/* Upload area */}
      <div className="upload-wrapper">
        <div style={{ paddingTop: 40 }}>
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
                {stages.map((s) => {
                  const isDone   = progress > s.threshold + 15;
                  const isActive = !isDone && progress >= s.threshold;
                  return (
                    <div
                      key={s.label}
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
      </div>
    </>
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

      const apiHost = process.env.NODE_ENV === "production" ? "" : "http://127.0.0.1:8000";
      const res = await fetch(`${apiHost}/analyze-stream`, {
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
              setResult({ alerts: safe.alerts || [], report: safe.report || "" });
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
    return <ResultSection result={result} elapsed={elapsed} onReset={handleReset} />;
  }

  return (
    <div className="app-shell">
      <BackgroundArt />
      <FloatingDots />
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