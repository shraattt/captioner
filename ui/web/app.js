/* Captioner browser UI: plain fetch calls against the local API.
   No build step and no framework, so the page works from the server as-is. */
"use strict";

const state = {
  jobs: [],
  detailId: null,
  token: localStorage.getItem("captioner.token") || "",
  // Display scale, so the page stays readable for anyone who needs it bigger.
  scale: Number(localStorage.getItem("captioner.scale")) || 1,
  choices: { spoken: [], translation: {} },
};
// The API pushes a snapshot when the queue changes; these pace the refetches
// that follow, so a burst of events cannot turn into a burst of requests.
const lastFetch = { jobs: 0, header: 0 };
const SCALES = [0.9, 1, 1.15, 1.3, 1.5, 1.75];

const $ = (id) => document.getElementById(id);

function toast(message, isError) {
  const box = $("toast");
  box.textContent = message;
  box.className = isError ? "error" : "";
  // Errors are announced immediately; ordinary progress stays polite so it
  // does not interrupt whatever the screen reader is reading.
  box.setAttribute("role", isError ? "alert" : "status");
  box.setAttribute("aria-live", isError ? "assertive" : "polite");
  box.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { box.hidden = true; }, isError ? 7000 : 3500);
}

function applyScale() {
  document.documentElement.style.setProperty("--scale", state.scale);
  $("text-size-value").textContent = Math.round(state.scale * 100) + "% text";
  $("text-normal").textContent = "A";
  localStorage.setItem("captioner.scale", state.scale);
}

function nudgeScale(step) {
  const index = SCALES.findIndex((value) => Math.abs(value - state.scale) < 0.01);
  const next = Math.min(SCALES.length - 1, Math.max(0, (index < 0 ? 1 : index) + step));
  state.scale = SCALES[next];
  applyScale();
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

function translationLabel(code) {
  const name = (state.choices.translation || {})[code];
  return name ? `${name} (${code})` : code;
}

function languageCell(job) {
  // What was heard and what it was turned into. The detected language is shown
  // even when you named one, so a guess that disagrees with the audio is
  // visible rather than invisible.
  const cell = document.createElement("td");
  cell.className = "small";
  const parts = [];
  if (job.source_lang_name || job.detected_lang_name) {
    parts.push(job.source_lang_name || job.detected_lang_name);
  }
  if (job.options && job.options.translate_to) {
    parts.push("\u2192 " + translationLabel(job.options.translate_to));
  }
  cell.textContent = parts.join(" ") || "\u2013";
  if (job.source_lang_name && job.detected_lang_name &&
      job.source_lang_name !== job.detected_lang_name) {
    cell.title = `heard ${job.detected_lang_name}, queued as ${job.source_lang_name}`;
  }
  return cell;
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

function button(text, handler, ghost, label) {
  const element = document.createElement("button");
  element.type = "button";
  if (ghost) element.className = "ghost";
  element.textContent = text;
  if (label) element.setAttribute("aria-label", label);
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

    const language = languageCell(job);

    const prog = document.createElement("td");
    prog.appendChild(progressCell(job));

    const created = document.createElement("td");
    created.className = "small";
    created.textContent = (job.created_at || "").slice(5, 16);
    created.title = job.created_at || "";

    const actions = document.createElement("td");
    const group = document.createElement("div");
    group.className = "actions";
    // Screen readers get the file name with each button, not just "view".
    const label = `job ${job.id} (${job.file_name})`;
    group.appendChild(button("view", () => showDetail(job.id), true, `view ${label}`));
    if (job.status === "queued" || job.status === "running") {
      group.appendChild(button("cancel", () => act("cancel", job.id), true,
                              `cancel ${label}`));
    } else {
      group.appendChild(button("retry", () => act("retry", job.id), true,
                              `retry ${label}`));
    }
    if (job.status !== "running") {
      group.appendChild(button("delete", () => act("delete", job.id), true,
                              `remove ${label}`));
    }
    actions.appendChild(group);

    row.append(id, file, status, model, language, prog, created, actions);
    body.appendChild(row);
  }
  if (!state.jobs.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 8;
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
      actions.appendChild(button("copy SRT", () => copyCaptions(job.id, "srt"), true,
                                 `copy the SRT of job ${job.id}`));
      actions.appendChild(button("copy plain text", () => copyCaptions(job.id, "text"), true,
                                 `copy the plain text of job ${job.id}`));
      actions.appendChild(button("translate\u2026", () => {
        $("manual-translate").open = true;
        $("detail-translate").focus();
      }, true, `translate job ${job.id}`));
    }

    const lines = job.caption_blocks.map(
      (block) => `[${(block.start_ms / 1000).toFixed(2)} - ${(block.end_ms / 1000).toFixed(2)}] ${block.text}`
    );
    const language = [
      job.source_lang_name ? `spoken:    ${job.source_lang_name}` : "",
      job.detected_lang_name && job.detected_lang_name !== job.source_lang_name
        ? `heard:     ${job.detected_lang_name}` : "",
      job.options && job.options.translate_to
        ? `translate: to ${translationLabel(job.options.translate_to)}` : "",
    ].filter(Boolean);
    const header = [
      `model:     ${job.model}`,
      ...language,
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
  $("detail-close").focus();
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

async function translateJob(jobId) {
  const target = $("detail-translate").value;
  if (!target) {
    toast("Choose a target language first", true);
    return;
  }
  const go = $("translate-go");
  go.disabled = true;
  toast(`Translating job #${jobId}\u2026`);
  try {
    const payload = await api(`/api/jobs/${jobId}/translate`, {
      method: "POST",
      body: JSON.stringify({ to: target }),
    });
    toast(`Wrote ${payload.files.join(", ")}`);
    await showDetail(jobId);
    await refreshJobs();
  } catch (error) {
    toast(error.message, true);
  } finally {
    go.disabled = false;
  }
}

async function refreshJobs() {
  lastFetch.jobs = Date.now();
  try {
    const payload = await api("/api/jobs?limit=200");
    state.jobs = payload.jobs;
    renderJobs();
    if (state.detailId) {
      // A job can be removed from the dashboard or the CLI while it is open
      // here; without this check the panel would keep asking for a 404 and
      // reporting an error every second.
      const stillThere = state.jobs.some((job) => job.id === state.detailId);
      if (stillThere) await showDetail(state.detailId);
      else {
        const gone = state.detailId;
        closeDetail();
        toast(`Job #${gone} is no longer in the queue`);
      }
    }
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

function fillLanguageSelects(spoken, translation) {
  // Spoken languages come from the recogniser, translation targets from the
  // translator: different sets, so the page lists each one truthfully.
  const speak = $("speak");
  speak.textContent = "";
  const detect = document.createElement("option");
  detect.value = "";
  detect.textContent = "detect automatically";
  speak.appendChild(detect);
  for (const [code, name] of Object.entries(spoken).sort((a, b) => a[1].localeCompare(b[1]))) {
    const option = document.createElement("option");
    option.value = code;
    option.textContent = `${name} (${code})`;
    speak.appendChild(option);
  }

  const targets = Object.entries(translation)
    .sort((a, b) => a[1].localeCompare(b[1]));
  for (const id of ["translate", "detail-translate"]) {
    const select = $(id);
    select.textContent = "";
    if (id === "translate") {
      const off = document.createElement("option");
      off.value = "";
      off.textContent = "(no translation)";
      select.appendChild(off);
    }
    for (const [code, name] of targets) {
      const option = document.createElement("option");
      option.value = code;
      option.textContent = `${name} (${code})`;
      select.appendChild(option);
    }
  }
  state.choices = { spoken, translation };
  renderJobs();  // the first paint may have run before the names arrived
}

async function fillSelects() {
  try {
    const payload = await api("/api/models");
    const model = $("model");
    model.textContent = "";
    for (const label of payload.batch) {
      const option = document.createElement("option");
      option.value = label;
      option.textContent = label;
      model.appendChild(option);
    }
    const languages = await api("/api/languages");
    fillLanguageSelects(languages.spoken || {}, languages.translation || languages.languages || {});
  } catch (error) {
    toast(error.message, true);
  }
}

$("queue-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const body = {
    path: $("path").value.trim(),
    model: $("model").value || undefined,
    // Empty means "detect it" and "do not translate" respectively.
    language: $("speak").value || undefined,
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
$("translate-go").addEventListener("click", () => {
  if (state.detailId) translateJob(state.detailId);
});
$("text-smaller").addEventListener("click", () => nudgeScale(-1));
$("text-larger").addEventListener("click", () => nudgeScale(1));
$("text-normal").addEventListener("click", () => {
  state.scale = 1;
  applyScale();
});
applyScale();

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
