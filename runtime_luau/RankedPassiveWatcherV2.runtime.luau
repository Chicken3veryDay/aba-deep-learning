-- ABA Ranked Passive Watcher v0.2.3
-- Passive only: no UI creation, no input simulation, no remote calls.
local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local UserInputService = game:GetService("UserInputService")
local HttpService = game:GetService("HttpService")
local Stats = game:GetService("Stats")

local G = getgenv()
local VERSION = "0.2.3-ranked-full-passive"
local SCHEMA_VERSION = "3.0.0"
local FEATURE_SCHEMA_ID = "ranked_explicit_v3_72"
local STREAM_FORMAT = "aba_ranked_passive_jsonl_v3"
local BASE_DIR = "aba_deep_learning/ranked_watch_v2"
local SOURCE_PATH = "aba_deep_learning/runtime_luau/RankedPassiveWatcherV2.runtime.luau"
local READY_PATH = BASE_DIR .. "/ready.json"
local SAMPLE_HZ = 15
local FLUSH_EVERY = 30
local MAX_STEPS = 18000
local OPPONENT_LOSS_GRACE_MS = 8000
local RESULT_SEAL_DELAY_MS = 1500

local FEATURE_NAMES = {
    "self_health_ratio",
    "target_health_ratio",
    "distance_norm_200",
    "relative_height_norm_50",
    "self_velocity_x_norm_100",
    "self_velocity_y_norm_100",
    "self_velocity_z_norm_100",
    "target_velocity_x_norm_100",
    "target_velocity_y_norm_100",
    "target_velocity_z_norm_100",
    "self_move_world_x",
    "self_move_world_z",
    "self_facing_dot_target",
    "self_facing_cross_target",
    "target_facing_dot_self",
    "target_facing_cross_self",
    "relative_direction_local_x",
    "relative_direction_local_z",
    "relative_velocity_local_x_norm_100",
    "relative_velocity_local_z_norm_100",
    "self_grounded",
    "target_grounded",
    "self_airborne",
    "target_airborne",
    "self_blocking_marker",
    "target_blocking_marker",
    "self_stunned_marker",
    "target_stunned_marker",
    "self_iframes_marker",
    "target_iframes_marker",
    "self_attacking_marker",
    "target_attacking_marker",
    "self_sprinting_estimate",
    "target_moving",
    "in_m1_range_6",
    "in_close_move_range_20",
    "target_on_screen",
    "target_screen_x",
    "target_screen_y",
    "camera_dot_self_forward",
    "camera_cross_self_forward",
    "camera_dot_target_direction",
    "camera_cross_target_direction",
    "camera_pitch",
    "camera_distance_to_self_norm_20",
    "ping_norm_250",
    "dt_norm_100ms",
    "self_health_delta_norm_20",
    "target_health_delta_norm_20",
    "self_took_damage",
    "target_took_damage",
    "self_state_running",
    "self_state_jumping",
    "self_state_freefall",
    "target_state_running",
    "target_state_jumping",
    "target_state_freefall",
    "self_animation_count_norm_12",
    "target_animation_count_norm_12",
    "self_walkspeed_norm_50",
    "target_walkspeed_norm_50",
    "self_jump_power_norm_100",
    "target_jump_power_norm_100",
    "camera_fov_norm_120",
    "camera_dot_self_forward_v2",
    "camera_cross_self_forward_v2",
    "camera_dot_target_direction_v2",
    "camera_cross_target_direction_v2",
    "camera_pitch_v2",
    "target_screen_x_v2",
    "target_screen_y_v2",
    "target_on_screen_v2",
}

local function nowMs()
    return DateTime.now().UnixTimestampMillis
end

local function finite(value)
    value = tonumber(value) or 0
    if value ~= value or value == math.huge or value == -math.huge then
        return 0
    end
    return value
end

local function clamp(value, minimum, maximum)
    return math.max(minimum, math.min(maximum, finite(value)))
end

local function norm(value, scale)
    if scale <= 0 then
        return 0
    end
    return clamp(finite(value) / scale, -1, 1)
end

local function bit(value)
    return value and 1 or 0
end

local function flatUnit(value)
    if typeof(value) ~= "Vector3" then
        return Vector3.new(0, 0, -1)
    end
    local flat = Vector3.new(value.X, 0, value.Z)
    if flat.Magnitude < 1e-5 then
        return Vector3.new(0, 0, -1)
    end
    return flat.Unit
end

local function vec3(value)
    if typeof(value) ~= "Vector3" then
        return {0, 0, 0}
    end
    return {finite(value.X), finite(value.Y), finite(value.Z)}
end

local function safeEnum(value)
    return tostring(value or "")
end

local function ensureFolders()
    if type(isfolder) ~= "function" or type(makefolder) ~= "function" then
        return
    end
    local parts = {
        "aba_deep_learning",
        "aba_deep_learning/runtime_luau",
        BASE_DIR,
    }
    for _, path in ipairs(parts) do
        if not isfolder(path) then
            makefolder(path)
        end
    end
end

local function appendText(path, text)
    if type(appendfile) == "function" then
        appendfile(path, text)
        return
    end
    local existing = ""
    if type(isfile) == "function" and isfile(path) then
        existing = readfile(path)
    end
    writefile(path, existing .. text)
end

local function encodeLine(record)
    local ok, encoded = pcall(HttpService.JSONEncode, HttpService, record)
    if ok then
        return encoded .. "\n"
    end
    return HttpService:JSONEncode({
        record_type = "serialization_error",
        schema_version = SCHEMA_VERSION,
        timestamp_ms = nowMs(),
        error = tostring(encoded),
    }) .. "\n"
end

local function controllerContamination()
    local names = {
        "ABA_AUTOPLAY_V3",
        "ABA_CONTROLLER_V3",
        "ABA_NEURAL_POLICY_RUNTIME",
        "ABA_NEURAL_RUNTIME",
        "ABA_DEEP_AGENT",
        "ABA_POLICY_EXECUTOR",
    }
    for _, name in ipairs(names) do
        local value = G[name]
        if type(value) == "table" then
            if value.Enabled == true
                or value.Active == true
                or value.Running == true
                or value.Executing == true then
                return true, name
            end
        elseif value == true then
            return true, name
        end
    end
    return false, nil
end

local function disableKnownControllers()
    local names = {
        "ABA_AUTOPLAY_V3",
        "ABA_CONTROLLER_V3",
        "ABA_NEURAL_POLICY_RUNTIME",
        "ABA_NEURAL_RUNTIME",
        "ABA_DEEP_AGENT",
        "ABA_POLICY_EXECUTOR",
    }
    for _, name in ipairs(names) do
        local value = G[name]
        if type(value) == "table" then
            if type(value.SetEnabled) == "function" then
                pcall(value.SetEnabled, value, false)
            end
            if value.Enabled ~= nil then value.Enabled = false end
            if value.Active ~= nil then value.Active = false end
            if value.Running ~= nil then value.Running = false end
        end
    end
end

local function humanoidParts(model)
    if not model then
        return nil, nil
    end
    local root = model:FindFirstChild("HumanoidRootPart") or model.PrimaryPart
    local humanoid = model:FindFirstChildOfClass("Humanoid")
    return root, humanoid
end

local function playerModel(player)
    return player and player.Character or nil
end

local function chooseOpponent(localPlayer, selfRoot)
    local bestPlayer, bestModel, bestRoot, bestHumanoid, bestDistance
    for _, player in ipairs(Players:GetPlayers()) do
        if player ~= localPlayer then
            local model = playerModel(player)
            local root, humanoid = humanoidParts(model)
            if root and humanoid and humanoid.Health > 0 then
                local distance = selfRoot
                    and (root.Position - selfRoot.Position).Magnitude
                    or math.huge
                if not bestDistance or distance < bestDistance then
                    bestPlayer = player
                    bestModel = model
                    bestRoot = root
                    bestHumanoid = humanoid
                    bestDistance = distance
                end
            end
        end
    end
    return bestPlayer, bestModel, bestRoot, bestHumanoid, bestDistance
end

local function markerMap(model)
    local output = {}
    if not model then
        return output
    end
    local count = 0
    for name, value in pairs(model:GetAttributes()) do
        count += 1
        if count > 64 then break end
        output["@" .. tostring(name)] = value
    end
    for _, child in ipairs(model:GetChildren()) do
        if child:IsA("BoolValue")
            or child:IsA("NumberValue")
            or child:IsA("IntValue")
            or child:IsA("StringValue") then
            output[child.Name] = child.Value
        end
    end
    return output
end

local function markerFlag(markers, patterns)
    for name, value in pairs(markers or {}) do
        local lower = string.lower(tostring(name))
        for _, pattern in ipairs(patterns) do
            if string.find(lower, pattern, 1, true) then
                if value == true then return 1 end
                if type(value) == "number" and value ~= 0 then return 1 end
                if type(value) == "string" then
                    local normalized = string.lower(value)
                    if normalized ~= ""
                        and normalized ~= "false"
                        and normalized ~= "0"
                        and normalized ~= "none" then
                        return 1
                    end
                end
            end
        end
    end
    return 0
end

local function animationSnapshot(humanoid)
    local output = {}
    if not humanoid then
        return output
    end
    local animator = humanoid:FindFirstChildOfClass("Animator")
    if not animator then
        return output
    end
    local ok, tracks = pcall(animator.GetPlayingAnimationTracks, animator)
    if not ok then
        return output
    end
    for index, track in ipairs(tracks) do
        if index > 16 then break end
        local animationId = ""
        pcall(function()
            animationId = track.Animation and track.Animation.AnimationId or ""
        end)
        output[#output + 1] = {
            id = animationId,
            time = finite(track.TimePosition),
            speed = finite(track.Speed),
            weight = finite(track.WeightCurrent),
            priority = safeEnum(track.Priority),
            looped = track.Looped == true,
        }
    end
    return output
end

local function humanoidState(humanoid)
    if not humanoid then
        return ""
    end
    local ok, state = pcall(humanoid.GetState, humanoid)
    return ok and tostring(state) or ""
end

local function grounded(humanoid)
    if not humanoid then
        return false
    end
    local state = humanoidState(humanoid)
    if string.find(state, "Jumping", 1, true)
        or string.find(state, "Freefall", 1, true)
        or string.find(state, "FallingDown", 1, true) then
        return false
    end
    return humanoid.FloorMaterial ~= Enum.Material.Air
end

local function pingMs()
    local ok, value = pcall(function()
        return Stats.Network.ServerStatsItem["Data Ping"]:GetValue()
    end)
    return ok and finite(value) or 0
end

local function cameraSnapshot(selfRoot, targetRoot)
    local camera = workspace.CurrentCamera
    if not camera then
        return {
            position = {0, 0, 0},
            look = {0, 0, -1},
            right = {1, 0, 0},
            up = {0, 1, 0},
            fov = 70,
            viewport = {0, 0},
            mouse = {0, 0},
            target_screen = {0, 0, 0},
            target_on_screen = false,
            features = {1, 0, 1, 0, 0, 0, 0, 0},
        }
    end

    local mouse = UserInputService:GetMouseLocation()
    local targetScreen = Vector3.zero
    local onScreen = false
    if targetRoot then
        targetScreen, onScreen = camera:WorldToViewportPoint(targetRoot.Position)
    end
    local viewport = camera.ViewportSize
    local cameraForward = flatUnit(camera.CFrame.LookVector)
    local selfForward = selfRoot and flatUnit(selfRoot.CFrame.LookVector)
        or Vector3.new(0, 0, -1)
    local targetDirection = selfRoot and targetRoot
        and flatUnit(targetRoot.Position - selfRoot.Position)
        or Vector3.new(0, 0, -1)
    local screenX = viewport.X > 0
        and clamp(targetScreen.X / viewport.X * 2 - 1, -1, 1)
        or 0
    local screenY = viewport.Y > 0
        and clamp(targetScreen.Y / viewport.Y * 2 - 1, -1, 1)
        or 0

    return {
        position = vec3(camera.CFrame.Position),
        look = vec3(camera.CFrame.LookVector),
        right = vec3(camera.CFrame.RightVector),
        up = vec3(camera.CFrame.UpVector),
        fov = finite(camera.FieldOfView),
        viewport = {finite(viewport.X), finite(viewport.Y)},
        mouse = {finite(mouse.X), finite(mouse.Y)},
        target_screen = {
            finite(targetScreen.X),
            finite(targetScreen.Y),
            finite(targetScreen.Z),
        },
        target_on_screen = onScreen == true,
        features = {
            clamp(cameraForward:Dot(selfForward), -1, 1),
            clamp(cameraForward:Cross(selfForward).Y, -1, 1),
            clamp(cameraForward:Dot(targetDirection), -1, 1),
            clamp(cameraForward:Cross(targetDirection).Y, -1, 1),
            clamp(camera.CFrame.LookVector.Y, -1, 1),
            screenX,
            screenY,
            onScreen and 1 or 0,
        },
    }
end

local HUD_NOISE_EXACT = {
    ["label"] = true,
    [">"] = true,
    ["<"] = true,
    ["x"] = true,
    ["current"] = true,
    ["on"] = true,
    ["off"] = true,
    ["material textures"] = true,
    ["show roblox material textures"] = true,
    ["sprint mode"] = true,
    ["changes input needed to run"] = true,
    ["mode music"] = true,
    ["changes when character themes play"] = true,
}

local function hudTextAllowed(path, text)
    local lowerPath = string.lower(tostring(path or ""))
    local lowerText = string.lower(
        (tostring(text or "")):gsub("^%s+", ""):gsub("%s+$", "")
    )
    if lowerText == "" or HUD_NOISE_EXACT[lowerText] then
        return false
    end
    if string.find(lowerPath, "watermark", 1, true)
        or string.find(lowerPath, "hidehudwarning", 1, true)
        or string.find(lowerPath, "settings", 1, true)
        or string.find(lowerPath, "options", 1, true)
        or string.find(lowerPath, "chat", 1, true) then
        return false
    end
    if string.find(lowerText, "snakeworl", 1, true)
        or string.find(lowerText, "game ui hidden", 1, true)
        or string.find(lowerText, "press h to undo", 1, true) then
        return false
    end
    return true
end

local function detectMatchResult(text)
    local normalized = string.lower(
        (tostring(text or "")):gsub("^%s+", ""):gsub("%s+$", "")
    )
    if normalized == "" then return nil end
    if string.find(normalized, "victory", 1, true)
        or string.find(normalized, "you win", 1, true)
        or string.find(normalized, "you won", 1, true) then
        return "victory"
    end
    if string.find(normalized, "defeat", 1, true)
        or string.find(normalized, "you lose", 1, true)
        or string.find(normalized, "you lost", 1, true) then
        return "defeat"
    end
    if normalized == "draw"
        or string.find(normalized, "match draw", 1, true)
        or string.find(normalized, "match tied", 1, true) then
        return "draw"
    end
    return nil
end

local function visibleHudText(localPlayer)
    local output = {}
    local matchResult = nil
    local playerGui = localPlayer and localPlayer:FindFirstChildOfClass("PlayerGui")
    if not playerGui then
        return output, matchResult
    end

    local count = 0
    for _, instance in ipairs(playerGui:GetDescendants()) do
        if (instance:IsA("TextLabel") or instance:IsA("TextButton"))
            and instance.Visible
            and instance.Text ~= "" then
            local path = instance:GetFullName()
            local text = tostring(instance.Text)
            matchResult = matchResult or detectMatchResult(text)
            if hudTextAllowed(path, text) then
                count += 1
                if count > 64 then break end
                output[#output + 1] = {
                    path = path,
                    text = text,
                    visible = true,
                    absolute_position = {
                        finite(instance.AbsolutePosition.X),
                        finite(instance.AbsolutePosition.Y),
                    },
                    absolute_size = {
                        finite(instance.AbsoluteSize.X),
                        finite(instance.AbsoluteSize.Y),
                    },
                }
            end
        end
    end
    return output, matchResult
end

local INPUT_KEYS = {
    [Enum.KeyCode.W] = "w",
    [Enum.KeyCode.A] = "a",
    [Enum.KeyCode.S] = "s",
    [Enum.KeyCode.D] = "d",
    [Enum.KeyCode.LeftShift] = "shift",
    [Enum.KeyCode.F] = "block",
    [Enum.KeyCode.Q] = "dodge",
    [Enum.KeyCode.Space] = "jump",
    [Enum.KeyCode.One] = "move1",
    [Enum.KeyCode.Two] = "move2",
    [Enum.KeyCode.Three] = "move3",
    [Enum.KeyCode.Four] = "move4",
    [Enum.KeyCode.G] = "transform",
}

local function inputName(input)
    if input.UserInputType == Enum.UserInputType.MouseButton1 then
        return "m1"
    end
    return INPUT_KEYS[input.KeyCode]
end

local function nearbyPlayers(localPlayer, selfRoot)
    local output = {}
    for _, player in ipairs(Players:GetPlayers()) do
        if player ~= localPlayer then
            local model = playerModel(player)
            local root, humanoid = humanoidParts(model)
            if root and humanoid then
                output[#output + 1] = {
                    name = player.Name,
                    display_name = player.DisplayName,
                    health = finite(humanoid.Health),
                    max_health = finite(humanoid.MaxHealth),
                    distance = selfRoot
                        and finite((root.Position - selfRoot.Position).Magnitude)
                        or 0,
                }
            end
        end
    end
    return output
end

local function featureVector(snapshot)
    local selfRoot = snapshot.self_root
    local selfHumanoid = snapshot.self_humanoid
    local targetRoot = snapshot.target_root
    local targetHumanoid = snapshot.target_humanoid
    local camera = snapshot.camera
    local selfMarkers = snapshot.self_markers
    local targetMarkers = snapshot.target_markers

    local selfHealth = selfHumanoid and selfHumanoid.Health or 0
    local selfMaxHealth = selfHumanoid and selfHumanoid.MaxHealth or 100
    local targetHealth = targetHumanoid and targetHumanoid.Health or 0
    local targetMaxHealth = targetHumanoid and targetHumanoid.MaxHealth or 100
    local selfVelocity = selfRoot and selfRoot.AssemblyLinearVelocity or Vector3.zero
    local targetVelocity = targetRoot and targetRoot.AssemblyLinearVelocity or Vector3.zero
    local offset = selfRoot and targetRoot
        and targetRoot.Position - selfRoot.Position
        or Vector3.zero
    local distance = offset.Magnitude
    local direction = flatUnit(offset)
    local selfForward = selfRoot and flatUnit(selfRoot.CFrame.LookVector)
        or Vector3.new(0, 0, -1)
    local targetForward = targetRoot and flatUnit(targetRoot.CFrame.LookVector)
        or Vector3.new(0, 0, -1)
    local selfLocalDirection = selfRoot
        and selfRoot.CFrame:VectorToObjectSpace(direction)
        or Vector3.zero
    local relativeVelocity = targetVelocity - selfVelocity
    local relativeLocalVelocity = selfRoot
        and selfRoot.CFrame:VectorToObjectSpace(relativeVelocity)
        or Vector3.zero
    local selfMove = selfHumanoid and selfHumanoid.MoveDirection or Vector3.zero
    local selfState = humanoidState(selfHumanoid)
    local targetState = humanoidState(targetHumanoid)
    local selfGrounded = grounded(selfHumanoid)
    local targetGrounded = grounded(targetHumanoid)
    local cameraPosition = camera.position
    local cameraWorldPosition = Vector3.new(
        cameraPosition[1] or 0,
        cameraPosition[2] or 0,
        cameraPosition[3] or 0
    )
    local cameraDistance = selfRoot
        and (cameraWorldPosition - selfRoot.Position).Magnitude
        or 0

    local values = {
        selfMaxHealth > 0 and clamp(selfHealth / selfMaxHealth, 0, 1) or 0,
        targetMaxHealth > 0 and clamp(targetHealth / targetMaxHealth, 0, 1) or 0,
        norm(distance, 200),
        norm(offset.Y, 50),
        norm(selfVelocity.X, 100),
        norm(selfVelocity.Y, 100),
        norm(selfVelocity.Z, 100),
        norm(targetVelocity.X, 100),
        norm(targetVelocity.Y, 100),
        norm(targetVelocity.Z, 100),
        clamp(selfMove.X, -1, 1),
        clamp(selfMove.Z, -1, 1),
        clamp(selfForward:Dot(direction), -1, 1),
        clamp(selfForward:Cross(direction).Y, -1, 1),
        clamp(targetForward:Dot(-direction), -1, 1),
        clamp(targetForward:Cross(-direction).Y, -1, 1),
        clamp(selfLocalDirection.X, -1, 1),
        clamp(selfLocalDirection.Z, -1, 1),
        norm(relativeLocalVelocity.X, 100),
        norm(relativeLocalVelocity.Z, 100),
        bit(selfGrounded),
        bit(targetGrounded),
        bit(not selfGrounded),
        bit(not targetGrounded),
        markerFlag(selfMarkers, {"block"}),
        markerFlag(targetMarkers, {"block"}),
        markerFlag(selfMarkers, {"stun", "hitstun"}),
        markerFlag(targetMarkers, {"stun", "hitstun"}),
        markerFlag(selfMarkers, {"iframe", "invulner", "immun"}),
        markerFlag(targetMarkers, {"iframe", "invulner", "immun"}),
        markerFlag(selfMarkers, {"attack", "busy", "usingmove"}),
        markerFlag(targetMarkers, {"attack", "busy", "usingmove"}),
        bit(selfHumanoid and selfHumanoid.WalkSpeed >= 24 and selfMove.Magnitude > 0.1),
        bit(targetHumanoid and targetHumanoid.MoveDirection.Magnitude > 0.1),
        bit(distance <= 6),
        bit(distance <= 20),
        bit(camera.target_on_screen),
        camera.features[6],
        camera.features[7],
        camera.features[1],
        camera.features[2],
        camera.features[3],
        camera.features[4],
        camera.features[5],
        norm(cameraDistance, 20),
        norm(snapshot.ping_ms, 250),
        norm(snapshot.dt_ms, 100),
        norm(snapshot.self_health_delta, 20),
        norm(snapshot.target_health_delta, 20),
        bit(snapshot.self_health_delta < 0),
        bit(snapshot.target_health_delta < 0),
        bit(string.find(selfState, "Running", 1, true)),
        bit(string.find(selfState, "Jumping", 1, true)),
        bit(string.find(selfState, "Freefall", 1, true)),
        bit(string.find(targetState, "Running", 1, true)),
        bit(string.find(targetState, "Jumping", 1, true)),
        bit(string.find(targetState, "Freefall", 1, true)),
        norm(#snapshot.self_animations, 12),
        norm(#snapshot.target_animations, 12),
        norm(selfHumanoid and selfHumanoid.WalkSpeed or 0, 50),
        norm(targetHumanoid and targetHumanoid.WalkSpeed or 0, 50),
        norm(selfHumanoid and selfHumanoid.JumpPower or 0, 100),
        norm(targetHumanoid and targetHumanoid.JumpPower or 0, 100),
        norm(camera.fov, 120),
        camera.features[1],
        camera.features[2],
        camera.features[3],
        camera.features[4],
        camera.features[5],
        camera.features[6],
        camera.features[7],
        camera.features[8],
    }

    assert(#values == 72, "ranked feature-vector width regression")
    return values
end

local existingWatcher = G.ABA_RANKED_PASSIVE_WATCHER_V2
if existingWatcher and type(existingWatcher.Status) == "function" then
    local ok, status = pcall(existingWatcher.Status, existingWatcher)
    if ok and status and (status.active or status.waiting) then
        return existingWatcher
    end
end

local bootToken = tostring(game.JobId)
    .. ":"
    .. tostring(nowMs())
    .. ":"
    .. tostring(math.random())
if G.ABA_RANKED_PASSIVE_WATCHER_V2_BOOT_LOCK ~= nil then
    return existingWatcher
end
G.ABA_RANKED_PASSIVE_WATCHER_V2_BOOT_LOCK = bootToken
task.delay(15, function()
    if G.ABA_RANKED_PASSIVE_WATCHER_V2_BOOT_LOCK == bootToken then
        G.ABA_RANKED_PASSIVE_WATCHER_V2_BOOT_LOCK = nil
    end
end)

local Watcher = {}
Watcher.__index = Watcher

function Watcher.new()
    return setmetatable({
        Version = VERSION,
        Active = false,
        Waiting = false,
        Started = false,
        Stopped = false,
        EpisodeId = nil,
        FilePath = nil,
        Header = nil,
        StepIndex = 0,
        Buffer = {},
        Connections = {},
        InputEvents = {},
        Held = {},
        LastSampleMs = nil,
        LastSelfHealth = nil,
        LastTargetHealth = nil,
        LastTargetSeenMs = nil,
        LastHudMs = 0,
        PendingMatchResult = nil,
        MatchResultSeenMs = nil,
        TargetPlayer = nil,
        TargetModel = nil,
        TargetRoot = nil,
        TargetHumanoid = nil,
    }, Watcher)
end

function Watcher:_connect(signal, callback)
    local connection = signal:Connect(callback)
    self.Connections[#self.Connections + 1] = connection
    return connection
end

function Watcher:_disconnectAll()
    for _, connection in ipairs(self.Connections) do
        pcall(function()
            connection:Disconnect()
        end)
    end
    table.clear(self.Connections)
end

function Watcher:_flush()
    if not self.FilePath or #self.Buffer == 0 then
        return
    end
    appendText(self.FilePath, table.concat(self.Buffer))
    table.clear(self.Buffer)
end

function Watcher:_write(record, force)
    self.Buffer[#self.Buffer + 1] = encodeLine(record)
    if force or #self.Buffer >= FLUSH_EVERY then
        self:_flush()
    end
end

function Watcher:_drainInputEvents()
    local events = self.InputEvents
    self.InputEvents = {}
    return events
end

function Watcher:_actionLabel(events)
    local label = {
        move_x = (self.Held.d and 1 or 0) - (self.Held.a and 1 or 0),
        move_z = (self.Held.w and 1 or 0) - (self.Held.s and 1 or 0),
        sprint = self.Held.shift and 1 or 0,
        block_held = self.Held.block == true,
        transform_held = self.Held.transform == true,
        m1 = false,
        dodge = false,
        jump = false,
        block_start = false,
        block_stop = false,
        move_slot = 0,
        source = "UserInputService",
    }
    for _, event in ipairs(events) do
        if event.phase == "began" then
            if event.name == "m1" then label.m1 = true end
            if event.name == "dodge" then label.dodge = true end
            if event.name == "jump" then label.jump = true end
            if event.name == "block" then label.block_start = true end
            if event.name == "move1" then label.move_slot = 1 end
            if event.name == "move2" then label.move_slot = 2 end
            if event.name == "move3" then label.move_slot = 3 end
            if event.name == "move4" then label.move_slot = 4 end
        elseif event.phase == "ended" and event.name == "block" then
            label.block_stop = true
        end
    end
    return label
end

function Watcher:_startEpisode(localPlayer, targetPlayer)
    ensureFolders()
    self.EpisodeId = "ranked-watch-v2-" .. tostring(nowMs())
    self.FilePath = BASE_DIR .. "/" .. self.EpisodeId .. ".jsonl"
    self.Header = {
        episode_id = self.EpisodeId,
        stream_format = STREAM_FORMAT,
        schema_version = SCHEMA_VERSION,
        feature_schema_id = FEATURE_SCHEMA_ID,
        feature_width = 72,
        feature_names = FEATURE_NAMES,
        watcher_version = VERSION,
        place_id = game.PlaceId,
        place_version = game.PlaceVersion,
        job_id = game.JobId,
        player = localPlayer.Name,
        opponent = targetPlayer and targetPlayer.Name or nil,
        started_at_ms = nowMs(),
        sample_hz = SAMPLE_HZ,
        observation_only = true,
        input_source = "UserInputService",
        executor_integration_allowed = false,
    }
    writefile(
        self.FilePath,
        encodeLine({
            record_type = "header",
            schema_version = SCHEMA_VERSION,
            header = self.Header,
        })
    )
    self.Active = true
    self.Waiting = false
    self.Started = true
    self.LastSampleMs = nil
    self.LastSelfHealth = nil
    self.LastTargetHealth = nil
    self.LastTargetSeenMs = nowMs()
end

function Watcher:_sample(localPlayer)
    if not self.Active then
        return
    end

    local contaminated, controllerName = controllerContamination()
    if contaminated then
        self:Stop("controller_contamination", {
            controller = controllerName,
            contaminated = true,
        })
        return
    end

    local character = localPlayer.Character
    local selfRoot, selfHumanoid = humanoidParts(character)
    if not selfRoot or not selfHumanoid then
        return
    end

    local targetPlayer, targetModel, targetRoot, targetHumanoid =
        chooseOpponent(localPlayer, selfRoot)
    if targetPlayer then
        self.TargetPlayer = targetPlayer
        self.TargetModel = targetModel
        self.TargetRoot = targetRoot
        self.TargetHumanoid = targetHumanoid
        self.LastTargetSeenMs = nowMs()
    else
        targetPlayer = self.TargetPlayer
        targetModel = self.TargetModel
        targetRoot = self.TargetRoot
        targetHumanoid = self.TargetHumanoid
    end

    local timestamp = nowMs()
    local dtMs = self.LastSampleMs and timestamp - self.LastSampleMs or 0
    self.LastSampleMs = timestamp

    local selfHealth = selfHumanoid.Health
    local targetHealth = targetHumanoid and targetHumanoid.Health or 0
    local selfHealthDelta = self.LastSelfHealth
        and selfHealth - self.LastSelfHealth
        or 0
    local targetHealthDelta = self.LastTargetHealth
        and targetHealth - self.LastTargetHealth
        or 0
    self.LastSelfHealth = selfHealth
    self.LastTargetHealth = targetHealth

    local camera = cameraSnapshot(selfRoot, targetRoot)
    local selfMarkers = markerMap(character)
    local targetMarkers = markerMap(targetModel)
    local selfAnimations = animationSnapshot(selfHumanoid)
    local targetAnimations = animationSnapshot(targetHumanoid)
    local events = self:_drainInputEvents()

    local snapshot = {
        self_root = selfRoot,
        self_humanoid = selfHumanoid,
        target_root = targetRoot,
        target_humanoid = targetHumanoid,
        camera = camera,
        self_markers = selfMarkers,
        target_markers = targetMarkers,
        self_animations = selfAnimations,
        target_animations = targetAnimations,
        ping_ms = pingMs(),
        dt_ms = dtMs,
        self_health_delta = selfHealthDelta,
        target_health_delta = targetHealthDelta,
    }

    local hud = nil
    if timestamp - self.LastHudMs >= 250 then
        hud, self.PendingMatchResult = visibleHudText(localPlayer)
        self.LastHudMs = timestamp
        if self.PendingMatchResult and not self.MatchResultSeenMs then
            self.MatchResultSeenMs = timestamp
        end
    end

    local observation = {
        schema_version = "1.0.0",
        episode_id = self.EpisodeId,
        step_index = self.StepIndex,
        timestamp_ms = timestamp,
        dt_ms = math.max(0, math.min(1000, dtMs)),
        self_state = {
            name = character.Name,
            health = finite(selfHealth),
            max_health = finite(selfHumanoid.MaxHealth),
            position = vec3(selfRoot.Position),
            velocity = vec3(selfRoot.AssemblyLinearVelocity),
            move_direction = vec3(selfHumanoid.MoveDirection),
            humanoid_state = humanoidState(selfHumanoid),
            markers = selfMarkers,
            animations = selfAnimations,
        },
        target_state = {
            name = targetPlayer and targetPlayer.Name or nil,
            display_name = targetPlayer and targetPlayer.DisplayName or nil,
            health = finite(targetHealth),
            max_health = finite(targetHumanoid and targetHumanoid.MaxHealth or 0),
            position = vec3(targetRoot and targetRoot.Position or Vector3.zero),
            velocity = vec3(
                targetRoot and targetRoot.AssemblyLinearVelocity or Vector3.zero
            ),
            humanoid_state = humanoidState(targetHumanoid),
            markers = targetMarkers,
            animations = targetAnimations,
            present = targetRoot ~= nil and targetHumanoid ~= nil,
        },
        relative = {
            distance = targetRoot
                and finite((targetRoot.Position - selfRoot.Position).Magnitude)
                or 0,
        },
        combat = {
            self_health_delta = finite(selfHealthDelta),
            target_health_delta = finite(targetHealthDelta),
        },
        cooldowns = {
            source = "hud_snapshot",
            hud = hud,
        },
        network = {
            ping_ms = snapshot.ping_ms,
        },
        history = {
            raw_input_events = events,
            held = self.Held,
        },
        camera = camera,
        nearby_players = nearbyPlayers(localPlayer, selfRoot),
        feature_schema_id = FEATURE_SCHEMA_ID,
        feature_vector = featureVector(snapshot),
    }

    self:_write({
        record_type = "step",
        schema_version = SCHEMA_VERSION,
        step = {
            observation = observation,
            action_label = self:_actionLabel(events),
            executor_result = {
                status = "not_applicable_passive_watcher",
                source = VERSION,
            },
            confirmations = {},
            raw_events = events,
            rewards = {
                damage_dealt = math.max(0, -targetHealthDelta),
                damage_received = math.max(0, -selfHealthDelta),
            },
        },
    })

    self.StepIndex += 1
    if self.StepIndex >= MAX_STEPS then
        self:Stop("max_steps")
        return
    end

    if self.MatchResultSeenMs
        and timestamp - self.MatchResultSeenMs >= RESULT_SEAL_DELAY_MS then
        self:Stop("match_result", {
            match_result = self.PendingMatchResult,
        })
        return
    end

    if self.Started
        and not targetPlayer
        and self.LastTargetSeenMs
        and timestamp - self.LastTargetSeenMs >= OPPONENT_LOSS_GRACE_MS then
        self:Stop("opponent_lost")
    end
end

function Watcher:Start()
    if self.Active or self.Waiting then
        return self:Status()
    end

    disableKnownControllers()
    local contaminated, controllerName = controllerContamination()
    if contaminated then
        error("controller remained active: " .. tostring(controllerName))
    end

    local localPlayer = Players.LocalPlayer
    if not localPlayer then
        error("Players.LocalPlayer is unavailable")
    end

    self.Waiting = true
    self:_connect(UserInputService.InputBegan, function(input, processed)
        local name = inputName(input)
        if not name then return end
        self.Held[name] = true
        self.InputEvents[#self.InputEvents + 1] = {
            timestamp_ms = nowMs(),
            phase = "began",
            name = name,
            processed = processed == true,
            key_code = tostring(input.KeyCode),
            input_type = tostring(input.UserInputType),
        }
    end)
    self:_connect(UserInputService.InputEnded, function(input, processed)
        local name = inputName(input)
        if not name then return end
        self.Held[name] = false
        self.InputEvents[#self.InputEvents + 1] = {
            timestamp_ms = nowMs(),
            phase = "ended",
            name = name,
            processed = processed == true,
            key_code = tostring(input.KeyCode),
            input_type = tostring(input.UserInputType),
        }
    end)

    local accumulator = 0
    self:_connect(RunService.Heartbeat, function(dt)
        if self.Stopped then return end
        local character = localPlayer.Character
        local selfRoot = character and humanoidParts(character)
        local targetPlayer = chooseOpponent(localPlayer, selfRoot)

        if self.Waiting and targetPlayer then
            self.TargetPlayer = targetPlayer
            self:_startEpisode(localPlayer, targetPlayer)
        end

        if self.Active then
            accumulator += dt
            local interval = 1 / SAMPLE_HZ
            if accumulator >= interval then
                accumulator %= interval
                local ok, err = pcall(function()
                    self:_sample(localPlayer)
                end)
                if not ok then
                    self:Stop("watcher_error", {error = tostring(err)})
                end
            end
        end
    end)

    ensureFolders()
    writefile(READY_PATH, HttpService:JSONEncode({
        version = VERSION,
        schema_version = SCHEMA_VERSION,
        feature_schema_id = FEATURE_SCHEMA_ID,
        source_path = SOURCE_PATH,
        job_id = game.JobId,
        place_id = game.PlaceId,
        waiting = true,
        timestamp_ms = nowMs(),
        passive = true,
        executor_integration_allowed = false,
    }))
    return self:Status()
end

function Watcher:Stop(reason, extra)
    if self.Stopped then
        return self:Status()
    end

    self.Active = false
    self.Waiting = false
    self.Stopped = true
    self:_disconnectAll()

    if self.FilePath then
        local terminal = {
            reason = reason or "stopped",
            match_result = self.PendingMatchResult,
            ended_at_ms = nowMs(),
            duration_ms = self.Header
                and math.max(0, nowMs() - self.Header.started_at_ms)
                or 0,
            steps = self.StepIndex,
            contaminated = reason == "controller_contamination",
            watcher_version = VERSION,
            executor_integration_allowed = false,
        }
        for key, value in pairs(extra or {}) do
            terminal[key] = value
        end
        self:_write({
            record_type = "terminal",
            schema_version = SCHEMA_VERSION,
            terminal = terminal,
        }, true)
    end

    if G.ABA_RANKED_PASSIVE_WATCHER_V2 == self then
        G.ABA_RANKED_PASSIVE_WATCHER_V2 = nil
    end
    G.ABA_RANKED_PASSIVE_WATCHER_V2_BOOT_LOCK = nil
    return self:Status()
end

function Watcher:Destroy()
    return self:Stop("destroyed")
end

function Watcher:Status()
    return {
        version = VERSION,
        active = self.Active,
        waiting = self.Waiting,
        stopped = self.Stopped,
        episode_id = self.EpisodeId,
        file_path = self.FilePath,
        steps = self.StepIndex,
        target = self.TargetPlayer and self.TargetPlayer.Name or nil,
        match_result = self.PendingMatchResult,
        passive = true,
        executor_integration_allowed = false,
    }
end

ensureFolders()
local watcher = Watcher.new()
G.ABA_RANKED_PASSIVE_WATCHER_V2 = watcher
G.ABA_RANKED_PASSIVE_WATCHER_V2_BOOT_LOCK = nil

local queue = queue_on_teleport or queueonteleport
if type(queue) == "function"
    and type(isfile) == "function"
    and isfile(SOURCE_PATH) then
    pcall(queue, "loadstring(readfile(" .. string.format("%q", SOURCE_PATH) .. "))()")
end

watcher:Start()
return watcher
