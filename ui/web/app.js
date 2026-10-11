/* Captioner browser UI: plain fetch calls against the local API.
   No build step and no framework, so the page works from the server as-is. */
"use strict";

const state = { jobs: [], detailId: null, token: localStorage.getItem("captioner.token") || "" };
// The API pushes a snapshot when the queue changes; these pace the refetches
// that follow, so a burst of events cannot turn into a burst of requests.
const lastFetch = { jobs: 0, header: 0 };

const $ = (id) => document.getElementById(id);

function toast(message, isError) {
  const box = $("toast");
  box.textContent = message;
  box.className = isError ? "error" : "";
  box.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { box.hidden = true; }, isError ? 7000 : 3500);
}

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json" };
  if (state.token) headers["Authorization"] = "Bearer " + state.token;
  const response = await fetch(path, { ...options, headers });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = (payload.error && payload.error.message) || response.statusText;
    throw new Error(detail);
  }
  return payload;
}

function pill(status) {
  const span = document.createElement("span");
  span.className = "status " + status;
  span.textContent = status;
  return span;
}

function progressCell(job) {
  if (job.status !== "running") {
    return document.createTextNode(job.error ? "error" : (job.stage || ""));
  }
  const bar = document.createElement("div");
  bar.className = "bar";
  const fill = document.createElement("i");
  const percent = Math.round((job.progress || 0) * 100);
  fill.style.width = percent + "%";
  bar.appendChild(fill);
  bar.title = percent + "% " + (job.stage || "");
  return bar;
}

function button(text, handler, ghost) {
  const element = document.createElement("button");
  if (ghost) element.className = "ghost";
  element.textContent = text;
  element.addEventListener("click", handler);
  return element;
}

function renderJobs() {
  const body = $("jobs").querySelector("tbody");
  body.textContent = "";
  for (const job of state.jobs) {
    const row = document.createElement("tr");

    const id = document.createElement("td");
    id.textContent = job.id;
    id.className = "small";

    const file = document.createElement("td");
    file.className = "file";
    file.textContent = job.file_name;
    file.title = job.file_path;

    const status = document.createElement("td");
    status.appendChild(pill(job.status));
    if (job.cancel_requested) {
      const stopping = document.createElement("span");
      stopping.className = "small";
      stopping.textContent = "stopping";
      stopping.title = "the worker stops at the next segment boundary";
      status.appendChild(stopping);
    }

    const model = document.createElement("td");
    model.className = "small";
    model.textContent = job.model;

    const prog = document.createElement("td");
    prog.appendChild(progressCell(job));

    const created = document.createElement("td");
    created.className = "small";
    created.textContent = (job.created_at || "").slice(5, 16);
    created.title = job.created_at || "";

    const actions = document.createElement("td");
    const group = document.createElement("div");
    group.className = "actions";
    group.appendChild(button("view", () => showDetail(job.id), true));
    if (job.status === "queued" || job.status === "running") {
      group.appendChild(button("cancel", () => act("cancel", job.id), true));
    } else {
      group.appendChild(button("retry", () => act("retry", job.id), true));
    }
    if (job.status !== "running") {
      group.appendChild(button("delete", () => act("delete", job.id), true));
    }
    actions.appendChild(group);

    row.append(id, file, status, model, prog, created, actions);
    body.appendChild(row);
  }
  if (!state.jobs.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 7;
    cell.className = "small";
    cell.textContent = "Nothing queued yet.";
    row.appendChild(cell);
    body.appendChild(row);
  }
}

async function act(action, jobId) {
  try {
    if (action === "cancel") {
      const payload = await api(`/api/jobs/${jobId}/cancel`, { method: "POST", body: "{}" });
      toast(payload.cancel === "requested"
        ? `Job #${jobId} will stop at the next segment boundary`
        : `Job #${jobId} canceled before it started`);
    } else if (action === "retry") {
      await api(`/api/jobs/${jobId}/retry`, { method: "POST", body: "{}" });
      toast(`Job #${jobId} requeued`);
    } else if (action === "delete") {
      await api(`/api/jobs/${jobId}`, { method: "DELETE" });
      toast(`Job #${jobId} removed`);
      if (state.detailId === jobId) closeDetail();
    }
    await refreshJobs();
  } catch (error) {
    toast(error.message, true);
  }
}

async function showDetail(jobId) {
  try {
    const payload = await api(`/api/jobs/${jobId}`);
    const job = payload.job;
    state.detailId = jobId;
    $("detail-panel").hidden = false;
    $("detail-title").textContent = `#${job.id} ${job.file_name} (${job.status})`;

    const actions = $("detail-actions");
    actions.textContent = "";
    if (job.status === "queued" || job.status === "running") {
      actions.appendChild(button("cancel", () => act("cancel", job.id)));
    } else {
      actions.appendChild(button("retry", () => act("retry", job.id)));
    }
    if (job.status !== "running") {
      actions.appendChild(button("delete", () => act("delete", job.id), true));
    }
    if (job.captions) {
      actions.appendChild(button("copy SRT", () => copyCaptions(job.id, "srt"), true));
      actions.appendChild(button("copy plain text", () => copyCaptions(job.id, "text"), true));
      actions.appendChild(button("translate…", () => promptTranslate(job.id), true));
    }

    const lines = job.caption_blocks.map(
      (block) => `[${(block.start_ms / 1000).toFixed(2)} - ${(block.end_ms / 1000).toFixed(2)}] ${block.text}`
    );
    const header = [
      `model:     ${job.model}`,
      `created:   ${job.created_at}`,
      `finished:  ${job.finished_at || "-"}`,
      `outputs:   ${job.outputs.join(", ") || "-"}`,
      `directory: ${job.output_dir || "-"}`,
      job.error ? `error:     ${job.error}` : "",
    ].filter(Boolean);
    $("detail-body").textContent = header.concat(["", ...lines]).join("\n");
  } catch (error) {
    toast(error.message, true);
  }
}

function closeDetail() {
  state.detailId = null;
  $("detail-panel").hidden = true;
}

async function copyCaptions(jobId, format) {
  try {
    const payload = await api(`/api/jobs/${jobId}/captions?format=${format}`);
    await navigator.clipboard.writeText(payload.text);
    toast(`Copied ${format} for job #${jobId}`);
  } catch (error) {
    toast(error.message, true);
  }
}

async function promptTranslate(jobId) {
  const target = window.prompt("Translate into which language code? (see /api/languages)", "hi");
  if (!target) return;
  toast(`Translating job #${jobId} into ${target}…`);
  try {
    const payload = await api(`/api/jobs/${jobId}/translate`, {
      method: "POST",
      body: JSON.stringify({ to: target }),
    });
    toast(`Wrote ${payload.files.join(", ")}`);
    await showDetail(jobId);
  } catch (error) {
    toast(error.message, true);
  }
}

async function refreshJobs() {
  lastFetch.jobs = Date.now();
  try {
    const payload = await api("/api/jobs?limit=200");
    state.jobs = payload.jobs;
    renderJobs();
    if (state.detailId) await showDetail(state.detailId);
  } catch (error) {
    toast(error.message, true);
  }
}

async function refreshHeader() {
  lastFetch.header = Date.now();
  try {
    const health = await api("/api/health");
    $("version").textContent = "v" + health.version;
    $("engine-asr").textContent = "ASR: " + health.engines.asr.selected;
    $("engine-mt").textContent = "MT: " + health.engines.mt.selected;
    const counts = Object.entries(health.queue).map(([k, v]) => `${k} ${v}`);
    $("queue").textContent = counts.length ? counts.join(" / ") : "queue empty";
    $("watch-note").textContent = "Watched folders: " +
      (health.watch_dirs.join(", ") || "none") +
      (health.webhook ? " - webhook enabled" : "");
  } catch (error) {
    toast("Cannot reach the API: " + error.message, true);
  }
}

async function fillSelects() {
  try {
    const payload = await api("/api/models");
    const model = $("model");
    for (const label of payload.batch) {
      const option = document.createElement("option");
      option.value = label;
      option.textContent = label;
      model.appendChild(option);
    }
    const languages = await api("/api/languages");
    const select = $("translate");
    for (const [code, name] of Object.entries(languages.languages)) {
      const option = document.createElement("option");
      option.value = code;
      option.textContent = `${name} (${code})`;
      select.appendChild(option);
    }
  } catch (error) {
    toast(error.message, true);
  }
}

$("queue-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const body = {
    path: $("path").value.trim(),
    model: $("model").value || undefined,
    translate_to: $("translate").value || undefined,
    diarize: $("diarize").checked,
    align: $("align").checked,
  };
  try {
    const payload = await api("/api/jobs", { method: "POST", body: JSON.stringify(body) });
    toast(`Queued job #${payload.job.id}`);
    $("path").value = "";
    await refreshJobs();
  } catch (error) {
    toast(error.message, true);
  }
});

$("detail-close").addEventListener("click", closeDetail);

$("token-save").addEventListener("click", () => {
  state.token = $("token").value.trim();
  if (state.token) localStorage.setItem("captioner.token", state.token);
  else localStorage.removeItem("captioner.token");
  toast(state.token ? "Token saved" : "Token cleared");
  refreshHeader();
  refreshJobs();
  connectEvents();  // the stream carries the token in its query string
});

if (state.token) $("token").value = state.token;

let events = null;

function connectEvents() {
  if (!window.EventSource) return;  // the slow poll below still updates the page
  if (events) events.close();
  const query = state.token ? "?token=" + encodeURIComponent(state.token) : "";
  events = new EventSource("/api/events" + query);
  events.onmessage = () => {
    // The snapshot only says something changed, so refetch what is on screen.
    if (Date.now() - lastFetch.jobs > 250) refreshJobs();
    if (Date.now() - lastFetch.header > 5000) refreshHeader();
  };
  // A dropped stream is not an error to report: EventSource reconnects on its
  // own, and the 30s poll keeps the page honest in the meantime.
  events.onerror = () => {};
}

refreshHeader();
refreshJobs();
fillSelects();
connectEvents();
setInterval(refreshJobs, 30000);  // fallback only: normally the stream drives this
setInterval(refreshHeader, 30000);
window.addEventListener("focus", () => refreshJobs());
