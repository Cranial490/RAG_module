const strategyOutput = document.getElementById("strategy-output");
const resultsOutput = document.getElementById("results-output");
const indexResultsOutput = document.getElementById("index-results-output");
const runButton = document.getElementById("run-button");
const indexButton = document.getElementById("index-button");
const retrievalStatusBadge = document.getElementById("status-badge");
const indexStatusBadge = document.getElementById("index-status-badge");
const retrievalTab = document.getElementById("retrieval-tab");
const indexingTab = document.getElementById("indexing-tab");
const retrievalPanel = document.getElementById("retrieval-panel");
const indexingPanel = document.getElementById("indexing-panel");

const formDefinitions = {
  retrieval: {
    configSelect: document.getElementById("config-select"),
    configInput: document.getElementById("config-input"),
    statusBadge: retrievalStatusBadge,
    strategySelects: {
      parser_key: document.getElementById("parser-select"),
      chunker_key: document.getElementById("chunker-select"),
      embedder_key: document.getElementById("embedder-select"),
      retrieval_key: document.getElementById("retrieval-select"),
      vector_db_key: document.getElementById("vector-db-select"),
    },
  },
  indexing: {
    configSelect: document.getElementById("index-config-select"),
    configInput: document.getElementById("index-config-input"),
    statusBadge: indexStatusBadge,
    strategySelects: {
      parser_key: document.getElementById("index-parser-select"),
      chunker_key: document.getElementById("index-chunker-select"),
      embedder_key: document.getElementById("index-embedder-select"),
      vector_db_key: document.getElementById("index-vector-db-select"),
    },
  },
};

const strategyResponseMap = {
  parser_key: "parsers",
  chunker_key: "chunkers",
  embedder_key: "embedders",
  retrieval_key: "retrievals",
  vector_db_key: "vector_dbs",
};

const retrievalQueryInput = document.getElementById("query-input");
const retrievalTopKInput = document.getElementById("top-k-input");
const filtersInput = document.getElementById("filters-input");
const metadataInput = document.getElementById("metadata-input");
const documentInput = document.getElementById("document-input");

let presets = [];
let currentStrategies = {};
const syncState = {
  retrieval: false,
  indexing: false,
};

function setStatus(badge, message, variant = "") {
  badge.textContent = message;
  badge.className = `status-badge ${variant}`.trim();
}

function formatJson(value) {
  return JSON.stringify(value, null, 2);
}

function parseJsonField(rawValue, fieldName) {
  const trimmed = rawValue.trim();
  if (!trimmed) {
    return null;
  }

  try {
    return JSON.parse(trimmed);
  } catch (error) {
    throw new Error(`${fieldName} must be valid JSON: ${error.message}`);
  }
}

function getSelectedPreset(formKey) {
  const form = formDefinitions[formKey];
  return presets[Number(form.configSelect.value)] ?? presets[0];
}

function fillSelectOptions(selectElement, values) {
  selectElement.innerHTML = "";

  const emptyOption = document.createElement("option");
  emptyOption.value = "";
  emptyOption.textContent = "None";
  selectElement.appendChild(emptyOption);

  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    selectElement.appendChild(option);
  });
}

function populateStrategySelects(formKey) {
  const form = formDefinitions[formKey];
  Object.entries(form.strategySelects).forEach(([configKey, selectElement]) => {
    const strategyGroup = currentStrategies[strategyResponseMap[configKey]];
    fillSelectOptions(selectElement, strategyGroup?.strategies ?? []);
  });
}

function syncSelectionsFromConfig(formKey, config) {
  const form = formDefinitions[formKey];
  syncState[formKey] = true;

  Object.entries(form.strategySelects).forEach(([configKey, selectElement]) => {
    const selectedValue = typeof config?.[configKey] === "string" ? config[configKey] : "";
    const allowedValues = currentStrategies[strategyResponseMap[configKey]]?.strategies ?? [];
    selectElement.value = allowedValues.includes(selectedValue) ? selectedValue : "";
  });

  syncState[formKey] = false;
}

function updateConfigEditorFromSelections(formKey) {
  const form = formDefinitions[formKey];
  let parsedConfig;

  try {
    parsedConfig = parseJsonField(form.configInput.value, "Config") ?? {};
  } catch {
    parsedConfig = {};
  }

  Object.entries(form.strategySelects).forEach(([configKey, selectElement]) => {
    if (selectElement.value) {
      parsedConfig[configKey] = selectElement.value;
    } else {
      delete parsedConfig[configKey];
    }
  });

  form.configInput.value = formatJson(parsedConfig);
}

function applyPresetToForm(formKey) {
  const form = formDefinitions[formKey];
  const preset = getSelectedPreset(formKey);
  const config = structuredClone(preset?.config ?? {});

  if (formKey === "indexing") {
    delete config.retrieval_key;
    delete config.retrieval_kwargs;
  }

  form.configInput.value = formatJson(config);
  syncSelectionsFromConfig(formKey, config);
}

function syncConfigFromEditor(formKey) {
  if (syncState[formKey]) {
    return;
  }

  const form = formDefinitions[formKey];

  try {
    const config = parseJsonField(form.configInput.value, "Config") ?? {};
    syncSelectionsFromConfig(formKey, config);
    setStatus(form.statusBadge, "Ready", "success");
  } catch {
    setStatus(form.statusBadge, "Invalid config JSON", "error");
  }
}

function initializeFormSelectors() {
  Object.keys(formDefinitions).forEach((formKey) => {
    const form = formDefinitions[formKey];

    populateStrategySelects(formKey);

    form.configSelect.innerHTML = "";
    presets.forEach((preset, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = preset.label;
      form.configSelect.appendChild(option);
    });

    applyPresetToForm(formKey);
  });
}

function activateTab(tabName) {
  const isRetrieval = tabName === "retrieval";
  retrievalTab.classList.toggle("active", isRetrieval);
  indexingTab.classList.toggle("active", !isRetrieval);
  retrievalPanel.classList.toggle("active", isRetrieval);
  indexingPanel.classList.toggle("active", !isRetrieval);
}

async function loadDashboardData() {
  setStatus(retrievalStatusBadge, "Loading strategies");
  setStatus(indexStatusBadge, "Loading strategies");

  const [strategyResponse, presetResponse] = await Promise.all([
    fetch("/strategies"),
    fetch("/dashboard/assets/presets.json"),
  ]);

  if (!strategyResponse.ok) {
    throw new Error(`Failed to load strategies: ${strategyResponse.status}`);
  }
  if (!presetResponse.ok) {
    throw new Error(`Failed to load presets: ${presetResponse.status}`);
  }

  currentStrategies = await strategyResponse.json();
  presets = await presetResponse.json();

  strategyOutput.textContent = formatJson(currentStrategies);
  initializeFormSelectors();
  setStatus(retrievalStatusBadge, "Ready", "success");
  setStatus(indexStatusBadge, "Ready", "success");
}

async function runRetrieval() {
  const query = retrievalQueryInput.value.trim();
  if (!query) {
    setStatus(retrievalStatusBadge, "Query is required", "error");
    resultsOutput.textContent = "Enter a query before running retrieval.";
    return;
  }

  let parsedConfig;
  let parsedFilters;

  try {
    updateConfigEditorFromSelections("retrieval");
    parsedConfig = parseJsonField(formDefinitions.retrieval.configInput.value, "Config") ?? {};
    parsedFilters = parseJsonField(filtersInput.value, "Filters");
  } catch (error) {
    setStatus(retrievalStatusBadge, "Invalid JSON", "error");
    resultsOutput.textContent = error.message;
    return;
  }

  setStatus(retrievalStatusBadge, "Running retrieval");
  resultsOutput.textContent = "Loading...";

  const response = await fetch(
    `/retrieve?query=${encodeURIComponent(query)}&top_k=${encodeURIComponent(retrievalTopKInput.value || "5")}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        config: parsedConfig,
        filters: parsedFilters,
      }),
    },
  );

  const body = await response.json().catch(() => ({}));
  resultsOutput.textContent = formatJson(body);

  if (!response.ok) {
    setStatus(retrievalStatusBadge, "Request failed", "error");
    return;
  }

  setStatus(
    retrievalStatusBadge,
    `Received ${Array.isArray(body) ? body.length : 0} result(s)`,
    "success",
  );
}

async function runIndexing() {
  const selectedFile = documentInput.files?.[0];
  if (!selectedFile) {
    setStatus(indexStatusBadge, "Document is required", "error");
    indexResultsOutput.textContent = "Choose a document before indexing.";
    return;
  }

  let parsedConfig;
  let parsedMetadata;

  try {
    updateConfigEditorFromSelections("indexing");
    parsedConfig = parseJsonField(formDefinitions.indexing.configInput.value, "Config") ?? {};
    parsedMetadata = parseJsonField(metadataInput.value, "Metadata");
  } catch (error) {
    setStatus(indexStatusBadge, "Invalid JSON", "error");
    indexResultsOutput.textContent = error.message;
    return;
  }

  const formData = new FormData();
  formData.append("config", JSON.stringify(parsedConfig));
  formData.append("file", selectedFile);
  if (parsedMetadata) {
    formData.append("metadata", JSON.stringify(parsedMetadata));
  }

  setStatus(indexStatusBadge, "Uploading document");
  indexResultsOutput.textContent = "Loading...";

  const response = await fetch("/index", {
    method: "POST",
    body: formData,
  });

  const body = await response.json().catch(() => ({}));
  indexResultsOutput.textContent = formatJson(body);

  if (!response.ok) {
    setStatus(indexStatusBadge, "Indexing failed", "error");
    return;
  }

  setStatus(
    indexStatusBadge,
    `Indexed ${body.chunks_indexed ?? 0} chunk(s)`,
    "success",
  );
}

retrievalTab.addEventListener("click", () => activateTab("retrieval"));
indexingTab.addEventListener("click", () => activateTab("indexing"));

Object.entries(formDefinitions).forEach(([formKey, form]) => {
  form.configSelect.addEventListener("change", () => applyPresetToForm(formKey));
  form.configInput.addEventListener("input", () => syncConfigFromEditor(formKey));

  Object.values(form.strategySelects).forEach((selectElement) => {
    selectElement.addEventListener("change", () => {
      updateConfigEditorFromSelections(formKey);
      setStatus(form.statusBadge, "Ready", "success");
    });
  });
});

runButton.addEventListener("click", () => {
  runRetrieval().catch((error) => {
    setStatus(retrievalStatusBadge, "Unexpected error", "error");
    resultsOutput.textContent = error.message;
  });
});

indexButton.addEventListener("click", () => {
  runIndexing().catch((error) => {
    setStatus(indexStatusBadge, "Unexpected error", "error");
    indexResultsOutput.textContent = error.message;
  });
});

activateTab("retrieval");
loadDashboardData().catch((error) => {
  setStatus(retrievalStatusBadge, "Failed to load dashboard", "error");
  setStatus(indexStatusBadge, "Failed to load dashboard", "error");
  strategyOutput.textContent = error.message;
  resultsOutput.textContent = error.message;
  indexResultsOutput.textContent = error.message;
});
