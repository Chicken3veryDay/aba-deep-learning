-- ABA Ranked Watcher Bootstrap v0.1.0
-- Installs one byte-pinned passive watcher. No UI, input simulation, or game remotes.
local HttpService = game:GetService("HttpService")

local G = getgenv()
local BOOTSTRAP_VERSION = "0.1.0"
local WATCHER_VERSION = "0.2.3-ranked-full-passive"
local WATCHER_COMMIT = "74b3b6278d25b97981dfdfe278a665525a2d608b"
local WATCHER_SHA256 = "d52b1795114dc287f1f91f33c487c96ad5f44f4694694b9202b0e8b75feff63f"
local WATCHER_BYTES = 39176
local REPOSITORY = "Chicken3veryDay/aba-deep-learning"
local SOURCE_PATH = "runtime_luau/RankedPassiveWatcherV2.runtime.luau"
local MANIFEST_PATH = "runtime_luau/RankedPassiveWatcherV2.manifest.json"
local LOCAL_SOURCE_PATH = "aba_deep_learning/runtime_luau/RankedPassiveWatcherV2.runtime.luau"
local RECEIPT_PATH = "aba_deep_learning/ranked_watch_v2/bootstrap_receipt.json"

local function ensureFolders()
    if type(isfolder) ~= "function" or type(makefolder) ~= "function" then
        error("filesystem folder functions are unavailable")
    end
    for _, path in ipairs({
        "aba_deep_learning",
        "aba_deep_learning/runtime_luau",
        "aba_deep_learning/ranked_watch_v2",
    }) do
        if not isfolder(path) then
            makefolder(path)
        end
    end
end

local function requestFunction()
    return request
        or http_request
        or (type(syn) == "table" and syn.request)
end

local function fetchRaw(path)
    local invoke = requestFunction()
    if type(invoke) ~= "function" then
        error("HTTP request function is unavailable")
    end
    local encodedPath = string.gsub(path, " ", "%%20")
    local url = "https://api.github.com/repos/"
        .. REPOSITORY
        .. "/contents/"
        .. encodedPath
        .. "?ref="
        .. WATCHER_COMMIT
    local response = invoke({
        Url = url,
        Method = "GET",
        Headers = {
            Accept = "application/vnd.github.raw+json",
            ["User-Agent"] = "ABA-Ranked-Watcher-Bootstrap",
        },
    })
    if type(response) ~= "table" or tonumber(response.StatusCode) ~= 200 then
        error(
            "GitHub fetch failed for "
                .. path
                .. " status="
                .. tostring(response and response.StatusCode)
        )
    end
    if type(response.Body) ~= "string" then
        error("GitHub response body is not text")
    end
    return response.Body
end

local function sha256(value)
    if type(crypt) ~= "table" or type(crypt.hash) ~= "function" then
        error("SHA-256 function is unavailable")
    end
    return string.lower(crypt.hash(value, "sha256"))
end

local function currentWatcher()
    local watcher = G.ABA_RANKED_PASSIVE_WATCHER_V2
    if type(watcher) ~= "table" or type(watcher.Status) ~= "function" then
        return nil, nil
    end
    local ok, status = pcall(watcher.Status, watcher)
    if not ok or type(status) ~= "table" then
        return watcher, nil
    end
    return watcher, status
end

local function stopSupersededWatcher()
    local watcher, status = currentWatcher()
    if not watcher then
        return
    end
    if status
        and status.version == WATCHER_VERSION
        and (status.active or status.waiting) then
        return watcher
    end
    if type(watcher.Stop) == "function" then
        pcall(watcher.Stop, watcher, "superseded_by_pinned_bootstrap")
    elseif type(watcher.Destroy) == "function" then
        pcall(watcher.Destroy, watcher)
    end
    G.ABA_RANKED_PASSIVE_WATCHER_V2 = nil
    G.ABA_RANKED_PASSIVE_WATCHER_V2_BOOT_LOCK = nil
end

ensureFolders()

local manifestText = fetchRaw(MANIFEST_PATH)
local okManifest, manifest = pcall(HttpService.JSONDecode, HttpService, manifestText)
if not okManifest or type(manifest) ~= "table" then
    error("watcher manifest is invalid JSON")
end
if manifest.version ~= WATCHER_VERSION then
    error("watcher manifest version mismatch")
end
if manifest.sha256 ~= WATCHER_SHA256 then
    error("watcher manifest SHA-256 mismatch")
end
if tonumber(manifest.bytes) ~= WATCHER_BYTES then
    error("watcher manifest byte count mismatch")
end
if manifest.executor_integration_allowed ~= false then
    error("watcher manifest does not prohibit executor integration")
end
if type(manifest.passive_invariants) ~= "table"
    or manifest.passive_invariants.creates_ui ~= false
    or manifest.passive_invariants.executes_remote_calls ~= false
    or manifest.passive_invariants.simulates_input ~= false then
    error("watcher manifest passive invariants failed")
end

local source = fetchRaw(SOURCE_PATH)
if #source ~= WATCHER_BYTES then
    error(
        "watcher source byte count mismatch expected="
            .. tostring(WATCHER_BYTES)
            .. " actual="
            .. tostring(#source)
    )
end
local sourceHash = sha256(source)
if sourceHash ~= WATCHER_SHA256 then
    error(
        "watcher source SHA-256 mismatch expected="
            .. WATCHER_SHA256
            .. " actual="
            .. sourceHash
    )
end

local compiled, compileError = loadstring(
    source,
    "RankedPassiveWatcherV2.runtime.luau"
)
if type(compiled) ~= "function" then
    error("watcher source failed to compile: " .. tostring(compileError))
end

writefile(LOCAL_SOURCE_PATH, source)
writefile(RECEIPT_PATH, HttpService:JSONEncode({
    bootstrap_version = BOOTSTRAP_VERSION,
    watcher_version = WATCHER_VERSION,
    watcher_commit = WATCHER_COMMIT,
    watcher_sha256 = WATCHER_SHA256,
    watcher_bytes = WATCHER_BYTES,
    local_source_path = LOCAL_SOURCE_PATH,
    installed_at_ms = DateTime.now().UnixTimestampMillis,
    job_id = game.JobId,
    place_id = game.PlaceId,
    passive = true,
    executor_integration_allowed = false,
}))

local existing = stopSupersededWatcher()
if existing then
    return existing
end

local watcher = compiled()
if type(watcher) ~= "table" or type(watcher.Status) ~= "function" then
    error("watcher bootstrap did not return a watcher object")
end
local status = watcher:Status()
if status.version ~= WATCHER_VERSION then
    error("installed watcher version mismatch")
end
if status.passive ~= true or status.executor_integration_allowed ~= false then
    error("installed watcher passive invariants failed")
end
return watcher
