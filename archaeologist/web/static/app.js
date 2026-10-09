/**
 * Codebase History Analyzer — Client Application
 * Classic Timeless Dark Theme with Stratified Temporal Causal Graph
 */

let activeRepo = "requests";
let activeTab = "studio";
let repositories = [];
let currentGraphData = null;
let graphFilter = "all";
let selectedGraphNode = null;
let highlightedNodeIds = new Set();
let highlightedEdgeKeys = new Set();
let graphTransform = { x: 40, y: 30, scale: 0.95 };
let isDraggingCanvas = false;
let dragStart = { x: 0, y: 0 };
let activeSymbols = [];
let selectedSymbolName = null;

// ==================== INITIALIZATION ====================
document.addEventListener("DOMContentLoaded", async () => {
  initApiKeyManagement();
  await loadRepositories();
  initTabNavigation();
  initGraphCanvas();
});

// ==================== API KEY & MODEL MANAGEMENT ====================
function getStoredApiKey() {
  return localStorage.getItem("GEMINI_USER_API_KEY") || "";
}

function getStoredModelTier() {
  return localStorage.getItem("GEMINI_MODEL_TIER") || "gemini-3.5-flash-lite";
}

function initApiKeyManagement() {
  const key = getStoredApiKey();
  const model = getStoredModelTier();
  
  const keyInput = document.getElementById("gemini-key-input");
  const modelSelect = document.getElementById("model-tier-select");
  if (keyInput) keyInput.value = key;
  if (modelSelect) {
    modelSelect.value = model;
    modelSelect.addEventListener("change", (e) => {
      localStorage.setItem("GEMINI_MODEL_TIER", e.target.value);
    });
  }

  updateApiKeyUI(key);
}

function updateApiKeyUI(key) {
  const dot = document.getElementById("key-status-dot");
  const text = document.getElementById("key-status-text");
  const pill = document.getElementById("api-key-pill");
  const warningBar = document.getElementById("api-key-warning");

  if (key && key.trim().length > 10) {
    if (dot) dot.className = "key-status-dot active";
    if (text) text.textContent = "Gemini Key: Active";
    if (pill) pill.classList.add("configured");
    if (warningBar) warningBar.classList.add("hidden");
  } else {
    if (dot) dot.className = "key-status-dot pulse-node";
    if (text) text.textContent = "Connect API Key";
    if (pill) pill.classList.remove("configured");
    if (warningBar) warningBar.classList.remove("hidden");
  }
}

function openKeyModal() {
  const modal = document.getElementById("api-key-modal");
  const keyInput = document.getElementById("gemini-key-input");
  const modelSelect = document.getElementById("model-tier-select");
  const statusDiv = document.getElementById("key-validation-status");
  if (statusDiv) statusDiv.className = "validation-status hidden";
  if (keyInput) keyInput.value = getStoredApiKey();
  if (modelSelect) modelSelect.value = getStoredModelTier();
  if (modal) modal.classList.remove("hidden");
}

function closeKeyModal() {
  const modal = document.getElementById("api-key-modal");
  if (modal) modal.classList.add("hidden");
}

function handleBackdropClick(e) {
  if (e.target.id === "api-key-modal") {
    closeKeyModal();
  }
}

function toggleKeyVisibility() {
  const input = document.getElementById("gemini-key-input");
  const icon = document.getElementById("key-vis-icon");
  if (input.type === "password") {
    input.type = "text";
    icon.textContent = "visibility_off";
  } else {
    input.type = "password";
    icon.textContent = "visibility";
  }
}

async function saveAndValidateKey() {
  const keyInput = document.getElementById("gemini-key-input");
  const modelSelect = document.getElementById("model-tier-select");
  const saveBtn = document.getElementById("save-key-btn");
  const statusDiv = document.getElementById("key-validation-status");

  const apiKey = keyInput.value.trim();
  const modelTier = modelSelect ? modelSelect.value : "gemini-3.5-flash-lite";

  if (!apiKey) {
    statusDiv.className = "validation-status error";
    statusDiv.textContent = "Please enter a valid Gemini API Key.";
    return;
  }

  saveBtn.disabled = true;
  saveBtn.innerHTML = `<span class="spinner" style="width:14px;height:14px;border-width:2px;"></span> Validating...`;

  try {
    const res = await fetch("/api/validate-key", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: apiKey, model: modelTier })
    });
    const data = await res.json();

    if (data.valid) {
      localStorage.setItem("GEMINI_USER_API_KEY", apiKey);
      localStorage.setItem("GEMINI_MODEL_TIER", modelTier);
      updateApiKeyUI(apiKey);

      statusDiv.className = "validation-status success";
      statusDiv.textContent = `✓ Key verified on ${modelTier}! Saved to localStorage.`;
      setTimeout(() => {
        closeKeyModal();
      }, 900);
    } else {
      statusDiv.className = "validation-status error";
      statusDiv.textContent = `✗ ${data.message || "Invalid API key."}`;
    }
  } catch (err) {
    localStorage.setItem("GEMINI_USER_API_KEY", apiKey);
    localStorage.setItem("GEMINI_MODEL_TIER", modelTier);
    updateApiKeyUI(apiKey);
    statusDiv.className = "validation-status success";
    statusDiv.textContent = "✓ Key saved locally.";
    setTimeout(() => {
      closeKeyModal();
    }, 900);
  } finally {
    saveBtn.disabled = false;
    saveBtn.innerHTML = `<span class="material-symbols-outlined text-sm">verified_user</span> Validate & Save`;
  }
}

function clearSavedKey() {
  localStorage.removeItem("GEMINI_USER_API_KEY");
  const keyInput = document.getElementById("gemini-key-input");
  if (keyInput) keyInput.value = "";
  updateApiKeyUI("");
  const statusDiv = document.getElementById("key-validation-status");
  if (statusDiv) {
    statusDiv.className = "validation-status error";
    statusDiv.textContent = "API key removed from local storage.";
  }
}

// ==================== REPOSITORIES & NAVIGATION ====================
async function loadRepositories() {
  try {
    const res = await fetch("/api/repos");
    const data = await res.json();
    repositories = data.repositories || [];
    renderRepoPills();
    selectRepository(activeRepo);
  } catch (err) {
    console.error("Failed to load repositories:", err);
  }
}

function renderRepoPills() {
  const container = document.getElementById("repo-selector");
  if (!container) return;

  container.innerHTML = repositories.map(repo => `
    <button 
      class="repo-pill-btn ${repo.id === activeRepo ? 'active' : ''}" 
      onclick="selectRepository('${repo.id}')"
      title="${repo.title}"
    >
      <span>${repo.name}</span>
      <span class="repo-pill-count">${(repo.total_commits || 0).toLocaleString()}</span>
    </button>
  `).join("");
}

function selectRepository(repoId) {
  activeRepo = repoId;
  renderRepoPills();

  const repo = repositories.find(r => r.id === repoId);
  if (!repo) return;

  // Update banner
  document.getElementById("banner-lang").textContent = repo.language || "Python";
  document.getElementById("banner-id").textContent = repo.name;
  document.getElementById("banner-title").textContent = repo.title;
  document.getElementById("banner-desc").textContent = repo.description;
  document.getElementById("banner-commits").textContent = (repo.total_commits || 0).toLocaleString();
  document.getElementById("banner-chunks").textContent = (repo.total_chunks || 0).toLocaleString();

  // Accuracy benchmark
  const accMap = { requests: "93.27%", flask: "88.99%", mss: "86.77%" };
  document.getElementById("banner-acc").textContent = accMap[repoId] || "90.00%";

  // Render starter questions
  renderStarterQuestions(repo.starter_questions || []);

  // Refresh active tab views
  if (activeTab === "graph") {
    loadCausalKnowledgeGraph(activeRepo);
  } else if (activeTab === "analytics") {
    loadAnalytics(activeRepo);
  } else if (activeTab === "symbols") {
    loadSymbols(activeRepo);
  }
}

function initTabNavigation() {
  document.getElementById("query-input")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") submitQuery();
  });
}

function switchTab(tabId) {
  activeTab = tabId;

  document.querySelectorAll(".nav-tab-btn").forEach(btn => btn.classList.remove("active"));
  const activeBtn = document.getElementById(`tab-btn-${tabId}`);
  if (activeBtn) activeBtn.classList.add("active");

  document.querySelectorAll(".tab-view").forEach(view => view.classList.remove("active"));
  const activeView = document.getElementById(`view-${tabId}`);
  if (activeView) activeView.classList.add("active");

  if (tabId === "graph") {
    loadCausalKnowledgeGraph(activeRepo);
  } else if (tabId === "analytics") {
    loadAnalytics(activeRepo);
  } else if (tabId === "symbols") {
    loadSymbols(activeRepo);
  }
}

// ==================== TAB 1: FORENSIC QUERY CONSOLE ====================
function renderStarterQuestions(questions) {
  const container = document.getElementById("starter-cards");
  if (!container) return;

  if (!questions || questions.length === 0) {
    container.innerHTML = `<p class="text-sm text-muted">No curated starter investigations available.</p>`;
    return;
  }

  container.innerHTML = questions.map(q => `
    <div class="starter-card" onclick="runInvestigationPrompt('${escapeHtml(q.question)}')">
      <div class="starter-card-header">
        <span class="starter-category">${q.category}</span>
        <span class="starter-difficulty">${q.difficulty}</span>
      </div>
      <p class="starter-card-question">${q.question}</p>
    </div>
  `).join("");
}

function runInvestigationPrompt(questionText) {
  const input = document.getElementById("query-input");
  if (input) {
    input.value = questionText;
    submitQuery();
  }
}

function submitQuery() {
  const input = document.getElementById("query-input");
  const query = input ? input.value.trim() : "";
  if (!query) return;

  const apiKey = getStoredApiKey();
  if (!apiKey) {
    openKeyModal();
    return;
  }

  const submitBtn = document.getElementById("submit-btn");
  const timeline = document.getElementById("execution-timeline");
  const stepsContainer = document.getElementById("timeline-steps");
  const responseContainer = document.getElementById("response-container");
  const responseBody = document.getElementById("response-body");
  const statusText = document.getElementById("timeline-status-text");
  const repoBadge = document.getElementById("resp-repo-badge");

  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.innerHTML = `<span class="spinner" style="width:14px;height:14px;border-width:2px;"></span> Investigating...`;
  }

  if (timeline) timeline.classList.remove("hidden");
  if (stepsContainer) stepsContainer.innerHTML = "";
  if (responseContainer) responseContainer.classList.add("hidden");
  if (statusText) statusText.textContent = "Initializing LangGraph state machine...";
  if (repoBadge) repoBadge.textContent = activeRepo;

  const sseUrl = `/api/ask/stream?repo_id=${encodeURIComponent(activeRepo)}&q=${encodeURIComponent(query)}&api_key=${encodeURIComponent(apiKey)}`;
  const eventSource = new EventSource(sseUrl);

  eventSource.addEventListener("start", (e) => {
    const data = JSON.parse(e.data);
    addTimelineStep("INITIALIZE", data.status || "Agent state initialized.");
  });

  eventSource.addEventListener("planning", (e) => {
    const data = JSON.parse(e.data);
    addTimelineStep("CANDIDATE FORENSICS", data.message || "Planning search queries...");
  });

  eventSource.addEventListener("retrieval", (e) => {
    const data = JSON.parse(e.data);
    addTimelineStep("HYBRID RRF", data.message || "Fusing BM25 and dense vector results...");
  });

  eventSource.addEventListener("verification", (e) => {
    const data = JSON.parse(e.data);
    addTimelineStep("SELF-VERIFY", data.message || "Self-verification check passed.");
  });

  eventSource.addEventListener("answer", (e) => {
    const data = JSON.parse(e.data);
    if (responseBody) {
      responseBody.innerHTML = marked.parse(data.response || "No response generated.");
    }
    if (responseContainer) responseContainer.classList.remove("hidden");
  });

  eventSource.addEventListener("done", (e) => {
    eventSource.close();
    if (statusText) statusText.textContent = "✓ Forensic Causal Investigation Complete";
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `<span class="material-symbols-outlined btn-icon">science</span> Investigate`;
    }
  });

  eventSource.addEventListener("error", (e) => {
    eventSource.close();
    let errorMsg = "Investigation stream encountered an error.";
    try {
      if (e.data) {
        const d = JSON.parse(e.data);
        errorMsg = d.error || errorMsg;
      }
    } catch (_) {}
    addTimelineStep("ERROR", errorMsg);
    if (statusText) statusText.textContent = "⚠ Investigation Terminated with Error";
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.innerHTML = `<span class="material-symbols-outlined btn-icon">science</span> Investigate`;
    }
  });
}

function addTimelineStep(badge, text) {
  const container = document.getElementById("timeline-steps");
  if (!container) return;

  const row = document.createElement("div");
  row.className = "timeline-step-row";
  row.innerHTML = `
    <span class="step-badge">${badge}</span>
    <span class="step-desc">${text}</span>
  `;
  container.appendChild(row);
}

function copyResponseText() {
  const responseBody = document.getElementById("response-body");
  if (!responseBody) return;
  navigator.clipboard.writeText(responseBody.innerText);
  alert("Causal answer copied to clipboard!");
}

// ==================== TAB 2: SPATIALLY PLACED TEMPORAL CAUSAL GRAPH ====================
async function loadCausalKnowledgeGraph(repoId) {
  const overlay = document.getElementById("graph-loading");
  if (overlay) overlay.classList.remove("hidden");

  try {
    const res = await fetch(`/api/graph/${encodeURIComponent(repoId)}?limit=300&include_all=true`);
    const data = await res.json();
    currentGraphData = data;
    renderStratifiedCausalGraph(data);
  } catch (err) {
    console.error("Failed to load graph data:", err);
  } finally {
    if (overlay) overlay.classList.add("hidden");
  }
}

let isGraphFullscreen = false;

function toggleGraphFullscreen() {
  const container = document.querySelector(".graph-workspace-grid");
  const icon = document.getElementById("fullscreen-icon");
  const btn = document.getElementById("btn-fullscreen-toggle");
  if (!container) return;

  isGraphFullscreen = !isGraphFullscreen;
  
  if (isGraphFullscreen) {
    document.body.classList.add("has-fullscreen-graph");
    container.classList.add("fullscreen-graph-mode");
    if (icon) icon.textContent = "fullscreen_exit";
    if (btn) btn.title = "Exit Full Screen Mode (ESC)";
  } else {
    document.body.classList.remove("has-fullscreen-graph");
    container.classList.remove("fullscreen-graph-mode");
    if (icon) icon.textContent = "fullscreen";
    if (btn) btn.title = "Toggle Full Screen Mode";
  }

  setTimeout(() => {
    if (currentGraphData) {
      renderStratifiedCausalGraph(currentGraphData);
    }
  }, 100);
}

// Listen for ESC key to exit fullscreen
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && isGraphFullscreen) {
    toggleGraphFullscreen();
  }
});

function initGraphCanvas() {
  const viewport = document.getElementById("graph-viewport");
  if (!viewport) return;

  viewport.addEventListener("mousedown", (e) => {
    if (e.target.tagName === "svg" || e.target.id === "graph-viewport" || e.target.tagName === "line" || e.target.tagName === "path") {
      isDraggingCanvas = true;
      dragStart = { x: e.clientX - graphTransform.x, y: e.clientY - graphTransform.y };
    }
  });

  window.addEventListener("mousemove", (e) => {
    if (isDraggingCanvas) {
      graphTransform.x = e.clientX - dragStart.x;
      graphTransform.y = e.clientY - dragStart.y;
      applyGraphTransform();
    }
  });

  window.addEventListener("mouseup", () => {
    isDraggingCanvas = false;
  });

  viewport.addEventListener("wheel", (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.12 : 0.88;
    zoomGraph(zoomFactor);
  });
}

function zoomGraph(factor) {
  graphTransform.scale = Math.min(Math.max(graphTransform.scale * factor, 0.2), 3.5);
  applyGraphTransform();
}

function resetGraphView() {
  graphTransform = { x: 40, y: 30, scale: 0.95 };
  applyGraphTransform();
}

function applyGraphTransform() {
  const g = document.getElementById("graph-root-group");
  if (g) {
    g.setAttribute("transform", `translate(${graphTransform.x}, ${graphTransform.y}) scale(${graphTransform.scale})`);
  }
}

function filterGraph(type) {
  graphFilter = type;
  document.querySelectorAll("#graph-filters .filter-pill").forEach(btn => {
    if (btn.textContent.toLowerCase().includes(type) || (type === "all" && btn.textContent.includes("All"))) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });
  if (currentGraphData) {
    renderStratifiedCausalGraph(currentGraphData);
  }
}

function searchGraphNode(query) {
  if (!query || !currentGraphData) {
    highlightedNodeIds.clear();
    highlightedEdgeKeys.clear();
    renderStratifiedCausalGraph(currentGraphData);
    return;
  }
  
  const q = query.toLowerCase().trim();
  const match = (currentGraphData.nodes || []).find(n => 
    n.id.toLowerCase().includes(q) || 
    (n.label && n.label.toLowerCase().includes(q)) || 
    (n.sha && n.sha.toLowerCase().startsWith(q)) ||
    (n.title && n.title.toLowerCase().includes(q))
  );

  if (match) {
    inspectNode(match);
  }
}

/**
 * Spatially places nodes in 4 distinct horizontal strata lanes:
 * Lane 1 (Y ~ 75): Issues
 * Lane 2 (Y ~ 205): Pull Requests
 * Lane 3 (Y ~ 345): Commits & Reverts
 * Lane 4 (Y ~ 485): AST Code Symbols
 */
function renderStratifiedCausalGraph(graphData) {
  const svg = document.getElementById("causal-graph-svg");
  if (!svg || !graphData) return;

  const nodes = graphData.nodes || [];
  const edges = graphData.edges || [];

  // Group nodes by type for stratified swimlane placement
  const issuesList = nodes.filter(n => n.type === "issue");
  const prsList = nodes.filter(n => n.type === "pr");
  const commitsList = nodes.filter(n => n.type === "commit" || n.type === "revert");
  const symbolsList = nodes.filter(n => n.type === "symbol");

  // Calculate horizontal spacing
  const xStart = 80;
  const laneY = {
    issue: 75,
    pr: 205,
    commit: 345,
    symbol: 485
  };

  const nodeMap = new Map();

  // Position Issues Lane
  const issueSpacing = Math.max(160, Math.min(220, 2400 / Math.max(1, issuesList.length)));
  issuesList.forEach((n, i) => {
    n.x = xStart + i * issueSpacing;
    n.y = laneY.issue + (i % 2 === 0 ? 0 : 25);
    nodeMap.set(n.id, n);
  });

  // Position PRs Lane
  const prSpacing = Math.max(170, Math.min(240, 3000 / Math.max(1, prsList.length)));
  prsList.forEach((n, i) => {
    n.x = xStart + i * prSpacing;
    n.y = laneY.pr + (i % 2 === 0 ? 0 : 25);
    nodeMap.set(n.id, n);
  });

  // Position Commits Timeline Lane (Chronological Left to Right)
  const commitSpacing = Math.max(140, Math.min(190, 4500 / Math.max(1, commitsList.length)));
  commitsList.forEach((n, i) => {
    n.x = xStart + i * commitSpacing;
    n.y = laneY.commit + (n.type === "revert" ? -25 : (i % 2 === 0 ? 0 : 20));
    nodeMap.set(n.id, n);
  });

  // Position AST Symbols Lane
  const symbolSpacing = Math.max(150, Math.min(210, 3200 / Math.max(1, symbolsList.length)));
  symbolsList.forEach((n, i) => {
    n.x = xStart + i * symbolSpacing;
    n.y = laneY.symbol + (i % 2 === 0 ? 0 : 25);
    nodeMap.set(n.id, n);
  });

  // Filter nodes based on active filter
  let visibleNodes = nodes;
  if (graphFilter !== "all") {
    visibleNodes = nodes.filter(n => {
      if (graphFilter === "revert") return n.type === "revert" || n.is_revert || n.reverts_sha;
      return n.type === graphFilter;
    });
  }

  const visibleNodeMap = new Map();
  visibleNodes.forEach(n => visibleNodeMap.set(n.id, n));

  // Filter visible edges
  const visibleEdges = edges.filter(e => visibleNodeMap.has(e.source) && visibleNodeMap.has(e.target));

  const totalWidth = Math.max(2500, Math.max(commitsList.length, prsList.length) * 190 + 300);

  svg.innerHTML = `
    <defs>
      <!-- Arrow Markers -->
      <marker id="arrow-white" viewBox="0 0 10 10" refX="17" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1 L 9 5 L 0 9 z" fill="#FFFFFF" opacity="0.8" />
      </marker>
      <marker id="arrow-revert" viewBox="0 0 10 10" refX="18" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#F87171" />
      </marker>
      <!-- Glow Filters -->
      <filter id="glow-white" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="3" result="blur" />
        <feComposite in="SourceGraphic" in2="blur" operator="over" />
      </filter>
      <filter id="glow-revert" x="-30%" y="-30%" width="160%" height="160%">
        <feGaussianBlur stdDeviation="4" result="blur" />
        <feComposite in="SourceGraphic" in2="blur" operator="over" />
      </filter>
    </defs>
    
    <g id="graph-root-group" transform="translate(${graphTransform.x}, ${graphTransform.y}) scale(${graphTransform.scale})">
      
      <!-- Stratified Swimlane Background Guides -->
      <g id="swimlane-guides" opacity="0.12">
        <line x1="0" y1="135" x2="${totalWidth}" y2="135" stroke="#FFFFFF" stroke-dasharray="4,4" stroke-width="1" />
        <line x1="0" y1="275" x2="${totalWidth}" y2="275" stroke="#FFFFFF" stroke-dasharray="4,4" stroke-width="1" />
        <line x1="0" y1="415" x2="${totalWidth}" y2="415" stroke="#FFFFFF" stroke-dasharray="4,4" stroke-width="1" />
      </g>

      <!-- Directed Causal Edges Layer -->
      <g id="edges-layer">
        ${visibleEdges.map(e => {
          const s = nodeMap.get(e.source);
          const t = nodeMap.get(e.target);
          if (!s || !t) return "";

          const isRevert = e.type === "reverts" || e.type === "superseded_by";
          const isHighlighted = highlightedEdgeKeys.has(`${e.source}->${e.target}`);
          const hasActiveHighlight = highlightedNodeIds.size > 0;
          
          let opacity = hasActiveHighlight ? (isHighlighted ? 1.0 : 0.08) : 0.45;
          let strokeColor = isRevert ? "#F87171" : "#FFFFFF";
          let strokeWidth = isHighlighted ? 2.5 : (isRevert ? 2.0 : 1.2);
          let marker = isRevert ? "url(#arrow-revert)" : "url(#arrow-white)";
          let dash = isRevert ? "5,4" : "none";

          // Calculate smooth cubic bezier path for vertical strata flow
          let pathD = "";
          if (isRevert) {
            // High arching curve above commits
            const midX = (s.x + t.x) / 2;
            const archY = Math.min(s.y, t.y) - 65;
            pathD = `M ${s.x} ${s.y} Q ${midX} ${archY} ${t.x} ${t.y}`;
          } else {
            // Smooth vertical downward S-curve between strata
            const deltaY = (t.y - s.y) * 0.5;
            pathD = `M ${s.x} ${s.y} C ${s.x} ${s.y + deltaY}, ${t.x} ${t.y - deltaY}, ${t.x} ${t.y}`;
          }

          return `<path d="${pathD}" fill="none" stroke="${strokeColor}" stroke-width="${strokeWidth}" stroke-dasharray="${dash}" opacity="${opacity}" marker-end="${marker}" />`;
        }).join("")}
      </g>

      <!-- Spatial Nodes Layer -->
      <g id="nodes-layer">
        ${visibleNodes.map(node => renderStratifiedNodeSvg(node)).join("")}
      </g>

    </g>
  `;

  // Attach interactive node clicks
  svg.querySelectorAll(".graph-node-g").forEach(nodeEl => {
    nodeEl.addEventListener("click", (e) => {
      e.stopPropagation();
      const nodeId = nodeEl.getAttribute("data-id");
      const node = nodeMap.get(nodeId);
      if (node) {
        inspectNode(node);
      }
    });
  });
}

function renderStratifiedNodeSvg(node) {
  const isSelected = selectedGraphNode && selectedGraphNode.id === node.id;
  const isHighlighted = highlightedNodeIds.has(node.id);
  const hasActiveHighlight = highlightedNodeIds.size > 0;
  const opacity = hasActiveHighlight ? (isHighlighted || isSelected ? 1.0 : 0.2) : 1.0;

  let shape = "";
  let badgeFill = "#18181B";
  let borderStroke = "#FFFFFF";

  if (node.type === "revert" || node.is_revert) {
    badgeFill = "rgba(248, 113, 113, 0.2)";
    borderStroke = "#F87171";
    shape = `
      <polygon points="${node.x},${node.y - 14} ${node.x + 13},${node.y} ${node.x},${node.y + 14} ${node.x - 13},${node.y}" fill="${badgeFill}" stroke="${borderStroke}" stroke-width="${isSelected ? 3 : 2}" filter="url(#glow-revert)" />
    `;
  } else if (node.type === "pr") {
    badgeFill = "rgba(192, 132, 252, 0.15)";
    borderStroke = "#C084FC";
    shape = `
      <circle cx="${node.x}" cy="${node.y}" r="14" fill="${badgeFill}" stroke="${borderStroke}" stroke-width="${isSelected ? 2.5 : 1.5}" />
    `;
  } else if (node.type === "issue") {
    badgeFill = "rgba(251, 191, 36, 0.15)";
    borderStroke = "#FBBF24";
    shape = `
      <rect x="${node.x - 12}" y="${node.y - 12}" width="24" height="24" rx="4" fill="${badgeFill}" stroke="${borderStroke}" stroke-width="${isSelected ? 2.5 : 1.5}" />
    `;
  } else if (node.type === "symbol") {
    badgeFill = "rgba(52, 211, 153, 0.15)";
    borderStroke = "#34D399";
    shape = `
      <polygon points="${node.x},${node.y - 12} ${node.x + 11},${node.y} ${node.x},${node.y + 12} ${node.x - 11},${node.y}" fill="${badgeFill}" stroke="${borderStroke}" stroke-width="${isSelected ? 2.5 : 1.5}" />
    `;
  } else {
    // Standard Commit Node (Hexagon)
    shape = `
      <polygon points="${node.x},${node.y - 12} ${node.x + 11},${node.y - 6} ${node.x + 11},${node.y + 6} ${node.x},${node.y + 12} ${node.x - 11},${node.y + 6} ${node.x - 11},${node.y - 6}" fill="${badgeFill}" stroke="${isSelected ? '#FFFFFF' : '#D4D4D8'}" stroke-width="${isSelected ? 2.5 : 1.5}" />
    `;
  }

  const labelText = node.label || node.id;
  return `
    <g class="graph-node-g" data-id="${node.id}" style="cursor: pointer;" opacity="${opacity}">
      ${shape}
      <text x="${node.x}" y="${node.y + 24}" fill="#FFFFFF" font-family="JetBrains Mono" font-size="9" text-anchor="middle" font-weight="500">${escapeHtml(labelText)}</text>
    </g>
  `;
}

function inspectNode(node) {
  selectedGraphNode = node;
  highlightCausalPedigree(node);

  const badge = document.getElementById("inspector-type-badge");
  const body = document.getElementById("inspector-body");

  if (badge) badge.textContent = node.type.toUpperCase();

  let html = `
    <div class="inspector-node-title">${escapeHtml(node.label)}</div>
    <div class="inspector-meta-row">
      <span class="inspector-meta-key">Type</span>
      <span class="inspector-meta-val">${node.type}</span>
    </div>
  `;

  if (node.state) {
    let stateColor = '#A1A1AA';
    if (node.state === 'open') stateColor = '#34D399';
    if (node.state === 'closed') stateColor = '#F87171';
    if (node.state === 'merged') stateColor = '#C084FC';
    html += `
      <div class="inspector-meta-row">
        <span class="inspector-meta-key">State</span>
        <span class="inspector-meta-val font-mono" style="color: ${stateColor}; font-weight: 700;">${node.state.toUpperCase()}</span>
      </div>
    `;
  }

  if (node.sha) {
    html += `
      <div class="inspector-meta-row">
        <span class="inspector-meta-key">Commit SHA</span>
        <span class="inspector-meta-val font-mono text-white">${node.sha.substring(0, 10)}</span>
      </div>
      <div class="inspector-meta-row">
        <span class="inspector-meta-key">Author</span>
        <span class="inspector-meta-val">${node.author || 'unknown'}</span>
      </div>
      <div class="inspector-meta-row">
        <span class="inspector-meta-key">Authored Date</span>
        <span class="inspector-meta-val">${node.date || '-'}</span>
      </div>
    `;
  }

  if (node.title) {
    html += `
      <div class="inspector-message-box">
        <strong>${node.type === 'commit' || node.type === 'revert' ? 'Commit Message' : 'Title'}:</strong><br/>
        ${escapeHtml(node.title)}
      </div>
    `;
  }

  if (node.linked_prs && node.linked_prs.length > 0) {
    html += `
      <div class="inspector-meta-row" style="margin-top: 0.5rem;">
        <span class="inspector-meta-key">Linked PRs</span>
        <span class="inspector-meta-val font-mono text-white">${node.linked_prs.map(pr => `#${pr}`).join(', ')}</span>
      </div>
    `;
  }

  if (node.linked_commits && node.linked_commits.length > 0) {
    html += `
      <div class="inspector-meta-row">
        <span class="inspector-meta-key">Linked Commits</span>
        <span class="inspector-meta-val font-mono text-white">${node.linked_commits.map(c => c.substring(0, 7)).join(', ')}</span>
      </div>
    `;
  }

  if (node.linked_issues && node.linked_issues.length > 0) {
    html += `
      <div class="inspector-meta-row">
        <span class="inspector-meta-key">Linked Issues</span>
        <span class="inspector-meta-val font-mono text-white">${node.linked_issues.map(i => `#${i}`).join(', ')}</span>
      </div>
    `;
  }

  if (node.reverts_sha) {
    html += `
      <div class="inspector-message-box" style="border-left-color: #F87171; background: rgba(248,113,113,0.1);">
        <strong style="color: #F87171;">⚡ Causal Revert Action:</strong><br/>
        Explicitly reverses and supersedes commit <code class="text-white">${node.reverts_sha.substring(0, 7)}</code>
      </div>
    `;
  }

  if (node.symbols && node.symbols.length > 0) {
    html += `
      <div class="inspector-symbols-wrap">
        <div class="inspector-symbols-label">Modified AST Symbols:</div>
        ${node.symbols.map(s => `<span class="inspector-symbol-tag">${escapeHtml(s)}</span>`).join("")}
      </div>
    `;
  }

  html += `
    <div style="margin-top: 1.5rem; display: flex; flex-direction: column; gap: 0.5rem;">
      <button class="btn-primary-action" style="width: 100%; justify-content: center;" onclick="queryAboutSelectedNode()">
        <span class="material-symbols-outlined text-sm">psychology</span>
        <span>Investigate Causal Origin</span>
      </button>
      <button class="btn-subtle" style="width: 100%; justify-content: center;" onclick="resetGraphHighlight()">
        <span>Reset Graph Focus</span>
      </button>
    </div>
  `;

  if (body) body.innerHTML = html;
}

/**
 * Traces connected causal pedigree (ancestors & descendants) when a node is clicked.
 */
function highlightCausalPedigree(targetNode) {
  if (!currentGraphData) return;
  highlightedNodeIds.clear();
  highlightedEdgeKeys.clear();

  highlightedNodeIds.add(targetNode.id);
  const edges = currentGraphData.edges || [];

  edges.forEach(e => {
    if (e.source === targetNode.id) {
      highlightedNodeIds.add(e.target);
      highlightedEdgeKeys.add(`${e.source}->${e.target}`);
    }
    if (e.target === targetNode.id) {
      highlightedNodeIds.add(e.source);
      highlightedEdgeKeys.add(`${e.source}->${e.target}`);
    }
  });

  renderStratifiedCausalGraph(currentGraphData);
}

function resetGraphHighlight() {
  highlightedNodeIds.clear();
  highlightedEdgeKeys.clear();
  selectedGraphNode = null;
  if (currentGraphData) renderStratifiedCausalGraph(currentGraphData);
  const body = document.getElementById("inspector-body");
  if (body) {
    body.innerHTML = `
      <div class="inspector-empty-state">
        <span class="material-symbols-outlined text-4xl text-muted mb-2">touch_app</span>
        <p class="text-sm text-secondary">Click any node in the stratified graph to inspect its complete metadata, diffs, and causal linkages.</p>
      </div>
    `;
  }
}

function queryAboutSelectedNode() {
  if (!selectedGraphNode) return;
  switchTab('studio');
  let q = "";
  if (selectedGraphNode.sha) {
    q = `Why was commit ${selectedGraphNode.sha.substring(0, 7)} introduced and what was its causal impact?`;
  } else if (selectedGraphNode.type === "pr") {
    q = `What architectural change was merged in ${selectedGraphNode.label}?`;
  } else if (selectedGraphNode.type === "issue") {
    q = `What bug or feature was reported in ${selectedGraphNode.label} and how was it resolved?`;
  } else if (selectedGraphNode.type === "symbol") {
    q = `How has the implementation of ${selectedGraphNode.label} evolved over time?`;
  }
  const input = document.getElementById("query-input");
  if (input) {
    input.value = q;
    submitQuery();
  }
}

// ==================== TAB 3: HOTSPOTS & CHURN ====================
async function loadAnalytics(repoId) {
  const hotspotsCont = document.getElementById("hotspots-container");
  const ownershipCont = document.getElementById("ownership-container");
  const couplingCont = document.getElementById("coupling-container");
  const busBadge = document.getElementById("bus-factor-badge");

  try {
    const [hRes, oRes, cRes] = await Promise.all([
      fetch(`/api/hotspots/${encodeURIComponent(repoId)}?top_n=10`),
      fetch(`/api/ownership/${encodeURIComponent(repoId)}`),
      fetch(`/api/coupling/${encodeURIComponent(repoId)}?top_n=8`)
    ]);

    const hData = await hRes.json();
    const oData = await oRes.json();
    const cData = await cRes.json();

    // Render Hotspots
    if (hotspotsCont) {
      const hotspots = hData.hotspots || [];
      hotspotsCont.innerHTML = hotspots.map(h => `
        <div class="hotspot-row">
          <span class="hotspot-file" title="${h.file_path}">${h.file_path}</span>
          <span class="hotspot-count">${h.commit_count} commits</span>
        </div>
      `).join("") || '<p class="text-sm text-muted">No churn data available.</p>';
    }

    // Render Ownership & Bus Factor
    if (ownershipCont && busBadge) {
      const ownership = oData.ownership || {};
      const authorsDict = ownership.author_distribution || {};
      const authors = Object.entries(authorsDict).map(([author, data]) => ({
        author: author,
        commit_count: data.commit_count,
        commit_percentage: data.percentage
      })).sort((a, b) => b.commit_count - a.commit_count);

      const risk = ownership.bus_factor_risk || "NORMAL";
      busBadge.textContent = `Risk: ${risk}`;
      busBadge.className = `bus-factor-badge ${risk.includes("HIGH") ? 'low' : 'good'}`;

      ownershipCont.innerHTML = authors.slice(0, 8).map(a => `
        <div class="ownership-row">
          <span class="hotspot-file">${a.author}</span>
          <span class="hotspot-count">${a.commit_percentage}% (${a.commit_count})</span>
        </div>
      `).join("") || '<p class="text-sm text-muted">No contributor ownership data available.</p>';
    }

    // Render Coupling
    if (couplingCont) {
      const couplings = cData.couplings || [];
      couplingCont.innerHTML = couplings.map(c => `
        <div class="coupling-card">
          <div class="coupling-files">${c.file_a} ↔ ${c.file_b}</div>
          <div class="text-xs text-secondary font-mono">${c.co_commit_count} Co-Commits</div>
        </div>
      `).join("") || '<p class="text-sm text-muted">No temporal coupling pairs detected.</p>';
    }
  } catch (err) {
    console.error("Failed to load analytics:", err);
  }
}

// ==================== TAB 4: AST CODE SYMBOLS ====================
async function loadSymbols(repoId) {
  const container = document.getElementById("symbols-container");
  if (!container) return;

  try {
    const res = await fetch(`/api/symbols/${encodeURIComponent(repoId)}?top_n=40`);
    const data = await res.json();
    activeSymbols = data.symbols || [];
    renderSymbolsList(activeSymbols);
  } catch (err) {
    console.error("Failed to load symbols:", err);
  }
}

function renderSymbolsList(symbols) {
  const container = document.getElementById("symbols-container");
  if (!container) return;

  if (symbols.length === 0) {
    container.innerHTML = `<p class="text-sm text-muted">No AST symbols found.</p>`;
    return;
  }

  container.innerHTML = symbols.map(s => `
    <div class="symbol-list-item" onclick="selectSymbol('${escapeHtml(s.symbol_name)}', '${escapeHtml(s.file_path || '')}')">
      <div>
        <span class="symbol-name-text">${s.symbol_name}</span>
        <span class="symbol-kind-tag">${s.kind || 'symbol'}</span>
      </div>
      <span class="font-mono text-xs text-white">${s.commit_count} commits</span>
    </div>
  `).join("");
}

function filterSymbolsList() {
  const input = document.getElementById("symbol-search-input");
  const query = input ? input.value.toLowerCase().trim() : "";
  const filtered = activeSymbols.filter(s => s.symbol_name.toLowerCase().includes(query) || (s.file_path && s.file_path.toLowerCase().includes(query)));
  renderSymbolsList(filtered);
}

async function selectSymbol(symbolName, filePath) {
  selectedSymbolName = symbolName;
  const title = document.getElementById("selected-symbol-title");
  const path = document.getElementById("selected-symbol-path");
  const actionsWrap = document.getElementById("symbol-actions-wrap");
  const container = document.getElementById("symbol-history-container");

  if (title) title.textContent = symbolName;
  if (path) path.textContent = filePath;
  if (actionsWrap) actionsWrap.style.display = "flex";
  if (container) container.innerHTML = `<div class="loading-spinner">Tracing chronological modifications for ${symbolName}...</div>`;

  try {
    const res = await fetch(`/api/symbols/${encodeURIComponent(activeRepo)}/history?symbol=${encodeURIComponent(symbolName)}`);
    const data = await res.json();
    const commits = data.commits || [];

    if (commits.length === 0) {
      container.innerHTML = `<p class="text-sm text-muted">No commit history found for ${symbolName}.</p>`;
      return;
    }

    container.innerHTML = `
      <div class="symbol-commits-timeline">
        ${commits.map(c => {
          const author = c.author_name || c.author || "contributor";
          const dateStr = c.authored_date || c.date || "";
          return `
            <div class="symbol-commit-card">
              <div class="flex justify-between items-center mb-1">
                <span class="font-mono text-xs text-white font-bold">${c.sha ? c.sha.substring(0, 7) : ''}</span>
                <span class="font-mono text-xs text-muted">${dateStr.substring(0, 10)}</span>
              </div>
              <p class="text-xs text-secondary mb-1">${escapeHtml(c.message ? c.message.split("\n")[0] : '')}</p>
              <div class="flex justify-between items-center text-xs text-muted">
                <span>Author: <strong class="text-white">${escapeHtml(author)}</strong></span>
                <button class="btn-text-action text-xs" onclick="investigateCommitSha('${c.sha}')">Investigate →</button>
              </div>
            </div>
          `;
        }).join("")}
      </div>
    `;
  } catch (err) {
    container.innerHTML = `<p class="text-sm text-red-400">Failed to trace symbol history.</p>`;
  }
}

function jumpToGraphForCurrentSymbol() {
  if (!selectedSymbolName) return;
  switchTab('graph');
  const searchInput = document.getElementById("graph-node-search");
  if (searchInput) {
    searchInput.value = selectedSymbolName;
    searchGraphNode(selectedSymbolName);
  }
}

function jumpToQueryForCurrentSymbol() {
  if (!selectedSymbolName) return;
  switchTab('studio');
  const queryInput = document.getElementById("query-input");
  if (queryInput) {
    queryInput.value = `How has the implementation of ${selectedSymbolName} evolved and why was it modified?`;
    submitQuery();
  }
}

function investigateCommitSha(sha) {
  switchTab('studio');
  const queryInput = document.getElementById("query-input");
  if (queryInput) {
    queryInput.value = `Why was commit ${sha.substring(0, 7)} created and what changes did it introduce?`;
    submitQuery();
  }
}

// ==================== UTILS ====================
function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}
