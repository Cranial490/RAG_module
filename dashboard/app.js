const configSelect = document.getElementById("config-select");
const parserSelect = document.getElementById("parser-select");
const chunkerSelect = document.getElementById("chunker-select");
const embedderSelect = document.getElementById("embedder-select");
const retrievalSelect = document.getElementById("retrieval-select");
const vectorDbSelect = document.getElementById("vector-db-select");
const queryInput = document.getElementById("query-input");
const topKInput = document.getElementById("top-k-input");
const filtersInput = document.getElementById("filters-input");
const configInput = document.getElementById("config-input");
const strategyOutput = document.getElementById("strategy-output");
const resultsOutput = document.getElementById("results-output");
const runButton = document.getElementById("run-button");
const statusBadge = document.getElementById("status-badge");

const strategySelectMap = {
  parser_key: parserSelect,
  chunker_key: chunkerSelect,
  embedder_key: embedderSelect,
  retrieval_key: retrievalSelect,
  vector_db_key: vectorDbSelect,
};

const strategyResponseMap = {
  parser_key: "parsers",
  chunker_key: "chunkers",
  embedder_key: "embedders",
  retrieval_key: "retrievals",
  vector_db_key: "vector_dbs",
};

let presets = [];
let currentStrategies = {};
let isSyncingForm = false;

function setStatus(message, variant = "") {
  statusBadge.textContent = message;
  statusBadge.className = `status-badge ${variant}`.trim();
}

function formatJson(value) {
  return JSON.stringify(value, null, 2);
}

function getSelectedPreset() {
  return presets[Number(configSelect.value)] ?? presets[0];
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

function populateStrategySelects(strategies) {
  Object.entries(strategySelectMap).forEach(([configKey, selectElement]) => {
    const strategyGroup = strategies[strategyResponseMap[configKey]];
    fillSelectOptions(selectElement, strategyGroup?.strategies ?? []);
  });
}

function updateConfigEditorFromSelections() {
  let parsedConfig;

  try {
    parsedConfig = parseJsonField(configInput.value, "Config") ?? {};
  } catch {
    parsedConfig = {};
  }

  Object.entries(strategySelectMap).forEach(([configKey, selectElement]) => {
    if (selectElement.value) {
      parsedConfig[configKey] = selectElement.value;
    } else {
      delete parsedConfig[configKey];
    }
  });

  configInput.value = formatJson(parsedConfig);
}

function syncSelectionsFromConfig(config) {
  isSyncingForm = true;

  Object.entries(strategySelectMap).forEach(([configKey, selectElement]) => {
    const selectedValue = typeof config?.[configKey] === "string" ? config[configKey] : "";
    const availableValues = currentStrategies[strategyResponseMap[configKey]]?.strategies ?? [];
    selectElement.value = availableValues.includes(selectedValue) ? selectedValue : "";
  });

  isSyncingForm = false;
}

function applyPresetToForm() {
  const preset = getSelectedPreset();
  const config = structuredClone(preset?.config ?? {});
  configInput.value = formatJson(config);
  syncSelectionsFromConfig(config);
}

function syncConfigFromEditor() {
  if (isSyncingForm) {
    return;
  }

  try {
    const config = parseJsonField(configInput.value, "Config") ?? {};
    syncSelectionsFromConfig(config);
    setStatus("Ready", "success");
  } catch (error) {
    setStatus("Invalid config JSON", "error");
  }
}

async function loadDashboardData() {
  setStatus("Loading strategies");

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
  populateStrategySelects(currentStrategies);

  configSelect.innerHTML = "";
  presets.forEach((preset, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = preset.label;
    configSelect.appendChild(option);
  });

  applyPresetToForm();
  setStatus("Ready", "success");
}

async function runRetrieval() {
  const query = queryInput.value.trim();
  if (!query) {
    setStatus("Query is required", "error");
    resultsOutput.textContent = "Enter a query before running retrieval.";
    return;
  }

  let parsedConfig;
  let parsedFilters;

  try {
    updateConfigEditorFromSelections();
    parsedConfig = parseJsonField(configInput.value, "Config") ?? {};
    parsedFilters = parseJsonField(filtersInput.value, "Filters");
  } catch (error) {
    setStatus("Invalid JSON", "error");
    resultsOutput.textContent = error.message;
    return;
  }

  setStatus("Running retrieval");
  resultsOutput.textContent = "Loading...";

  const response = await fetch(
    `/retrieve?query=${encodeURIComponent(query)}&top_k=${encodeURIComponent(topKInput.value || "5")}`,
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
    setStatus("Request failed", "error");
    return;
  }

  setStatus(`Received ${Array.isArray(body) ? body.length : 0} result(s)`, "success");
}

configSelect.addEventListener("change", applyPresetToForm);
configInput.addEventListener("input", syncConfigFromEditor);
Object.values(strategySelectMap).forEach((selectElement) => {
  selectElement.addEventListener("change", () => {
    updateConfigEditorFromSelections();
    setStatus("Ready", "success");
  });
});

runButton.addEventListener("click", () => {
  runRetrieval().catch((error) => {
    setStatus("Unexpected error", "error");
    resultsOutput.textContent = error.message;
  });
});

loadDashboardData().catch((error) => {
  setStatus("Failed to load dashboard", "error");
  strategyOutput.textContent = error.message;
  resultsOutput.textContent = error.message;
});
