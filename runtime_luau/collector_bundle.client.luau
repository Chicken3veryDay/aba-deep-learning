-- ABA Deep Learning Step 2 collector
-- Observation-only. This script never dispatches remotes or simulates input.

local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local UserInputService = game:GetService("UserInputService")
local HttpService = game:GetService("HttpService")
local Stats = game:GetService("Stats")

local LocalPlayer = Players.LocalPlayer
local VERSION = "0.2.0"
local SCHEMA_VERSION = "1.0.0"
local STREAM_FORMAT = "aba_episode_jsonl_v1"
local FEATURE_COUNT = 64

local function nowMs()
    return DateTime.now().UnixTimestampMillis
end

local function clamp(value, minimum, maximum)
    value = tonumber(value) or 0
    return math.max(minimum, math.min(maximum, value))
end

local function norm(value, scale)
    return clamp((tonumber(value) or 0) / scale, -1, 1)
end

local function bit(value)
    return value and 1 or 0
end

local function disconnectAll(connections)
    for _, connection in ipairs(connections) do
        pcall(function()
            connection:Disconnect()
        end)
    end
    table.clear(connections)
end

local function getHumanoid(model)
    return model and model:FindFirstChildOfClass("Humanoid") or nil
end

local function getRoot(model)
    if not model then
        return nil
    end
    return model:FindFirstChild("HumanoidRootPart")
        or model:FindFirstChild("Torso")
        or model.PrimaryPart
end

local function readValue(model, name, default)
    local value = model and model:FindFirstChild(name)
    if value and value:IsA("ValueBase") then
        return value.Value, true
    end
    return default, false
end

local function marker(model, names)
    if not model then
        return false, nil
    end
    for _, name in ipairs(names) do
        local instance = model:FindFirstChild(name, true)
        if instance then
            return true, instance:GetFullName()
        end
    end
    return false, nil
end

local function animationState(model)
    local humanoid = getHumanoid(model)
    local animator = humanoid and humanoid:FindFirstChildOfClass("Animator")
    if not animator then
        return nil
    end

    local selected = nil
    for _, track in ipairs(animator:GetPlayingAnimationTracks()) do
        if track.IsPlaying and (not selected or track.WeightCurrent >= selected.WeightCurrent) then
            selected = track
        end
    end
    if not selected then
        return nil
    end

    return {
        animation_id = selected.Animation and selected.Animation.AnimationId or "",
        time_position = selected.TimePosition,
        length = selected.Length,
        speed = selected.Speed,
        weight = selected.WeightCurrent,
        priority = tostring(selected.Priority),
    }
end

local function pingMs()
    local ok, result = pcall(function()
        local network = Stats:FindFirstChild("Network")
        local server = network and network:FindFirstChild("ServerStatsItem")
        local item = server and server:FindFirstChild("Data Ping")
        return item and item:GetValue() or 0
    end)
    return ok and tonumber(result) or 0
end

local function moveHud(slot)
    local gui = LocalPlayer and LocalPlayer:FindFirstChildOfClass("PlayerGui")
    local hud = gui and gui:FindFirstChild("HUD")
    local moves = hud and hud:FindFirstChild("Moves")
    local button = moves and moves:FindFirstChild(tostring(slot))
    if not button or not button:IsA("GuiObject") then
        return {
            ready = false,
            fill_ratio = 1,
            disabled = true,
            move_name = nil,
            source = "missing_hud",
            confidence = 0,
        }
    end

    local cooldown = button:FindFirstChild("CD")
    local cover = button:FindFirstChild("DisableCover")
    local fill = cooldown and cooldown:IsA("GuiObject") and clamp(cooldown.Size.Y.Scale, 0, 1) or 0
    local disabled = cover and cover:IsA("GuiObject") and cover.Visible or false
    return {
        ready = button.Visible and not disabled and fill <= 0.001,
        fill_ratio = fill,
        disabled = disabled,
        move_name = button:GetAttribute("MoveName"),
        source = "hud_cooldown_fill",
        confidence = 0.9,
    }
end

local function nearestTarget(character)
    local live = workspace:FindFirstChild("Live")
    local selfRoot = getRoot(character)
    if not live or not selfRoot then
        return nil
    end

    local selected = nil
    local selectedDistance = math.huge
    for _, candidate in ipairs(live:GetChildren()) do
        if candidate ~= character and candidate:IsA("Model") then
            local humanoid = getHumanoid(candidate)
            local root = getRoot(candidate)
            if humanoid and humanoid.Health > 0 and root then
                local distance = (root.Position - selfRoot.Position).Magnitude
                if distance < selectedDistance then
                    selected = candidate
                    selectedDistance = distance
                end
            end
        end
    end
    return selected
end

local function resolveTarget(character, explicit)
    if explicit and explicit.Parent then
        return explicit
    end
    local legacy = getgenv().ABA_AUTOPLAY_V3
    if legacy and legacy.Target and legacy.Target.Parent then
        return legacy.Target
    end
    return nearestTarget(character)
end

local function angleError(lookVector, direction)
    local a = Vector3.new(lookVector.X, 0, lookVector.Z)
    local b = Vector3.new(direction.X, 0, direction.Z)
    if a.Magnitude < 0.001 or b.Magnitude < 0.001 then
        return 0
    end
    return math.deg(math.acos(clamp(a.Unit:Dot(b.Unit), -1, 1)))
end

local function lineOfSight(character, target, root, targetRoot)
    if not root or not targetRoot then
        return false
    end
    local params = RaycastParams.new()
    params.FilterType = Enum.RaycastFilterType.Exclude
    params.FilterDescendantsInstances = character and { character } or {}
    params.IgnoreWater = true
    local result = workspace:Raycast(root.Position, targetRoot.Position - root.Position, params)
    return result == nil or (target and result.Instance:IsDescendantOf(target))
end

local function entityState(model, other)
    local humanoid = getHumanoid(model)
    local root = getRoot(model)
    local otherRoot = getRoot(other)
    local blocking, blockingKnown = readValue(model, "Blocking", false)
    local busy, busyKnown = readValue(model, "Busy", false)
    local combo, comboKnown = readValue(model, "Combo", 0)
    local stunned, stunMarker = marker(model, { "Stun", "Stunned", "Hitstun", "HitStun" })
    local ragdolled, ragdollMarker = marker(model, { "Ragdoll", "Ragdolled", "RagdollValue" })
    local recovering, recoveryMarker = marker(model, { "GettingUp", "GetUp", "Recovering", "Recovery" })
    local invulnerable, immunityMarker = marker(model, { "IFrame", "IFrames", "Invulnerable", "Immunity", "M1Immunity" })

    local health = humanoid and humanoid.Health or 0
    local maxHealth = humanoid and humanoid.MaxHealth or 100
    local velocity = root and root.AssemblyLinearVelocity or Vector3.zero
    local facing = 0
    if root and otherRoot then
        local direction = otherRoot.Position - root.Position
        local flat = Vector3.new(direction.X, 0, direction.Z)
        local look = Vector3.new(root.CFrame.LookVector.X, 0, root.CFrame.LookVector.Z)
        if flat.Magnitude > 0.001 and look.Magnitude > 0.001 then
            facing = clamp(look.Unit:Dot(flat.Unit), -1, 1)
        end
    end

    return {
        name = model and model.Name or nil,
        health = health,
        max_health = maxHealth,
        health_ratio = maxHealth > 0 and health / maxHealth or 0,
        speed_x = velocity.X,
        speed_z = velocity.Z,
        vertical_speed = velocity.Y,
        facing_target_cos = facing,
        grounded = humanoid and humanoid.FloorMaterial ~= Enum.Material.Air or false,
        airborne = humanoid and humanoid.FloorMaterial == Enum.Material.Air or true,
        stunned = stunned,
        ragdolled = ragdolled,
        blocking = blocking == true,
        recovering = recovering,
        invulnerable = invulnerable,
        action_active = busy == true,
        action_progress = 0,
        m1_stage = tonumber(combo) or 0,
        m1_reset_remaining_ms = 0,
        action_lock_remaining_ms = 0,
        state_confidence = (blockingKnown or busyKnown or comboKnown) and 0.85 or 0.55,
        current_animation = animationState(model),
        raw_signals = {
            blocking_known = blockingKnown,
            busy_known = busyKnown,
            combo_known = comboKnown,
            stun_marker = stunMarker,
            ragdoll_marker = ragdollMarker,
            recovery_marker = recoveryMarker,
            immunity_marker = immunityMarker,
        },
    }
end

local function featureVector(state)
    local selfState = state.self_state
    local target = state.target_state
    local relative = state.relative
    local combat = state.combat
    local cooldowns = state.cooldowns
    local network = state.network
    local history = state.history

    local vector = {
        clamp(selfState.health_ratio, 0, 1),
        norm(selfState.speed_x, 40), norm(selfState.speed_z, 40), norm(selfState.vertical_speed, 80),
        clamp(selfState.facing_target_cos, -1, 1), bit(selfState.grounded), bit(selfState.airborne),
        bit(selfState.stunned), bit(selfState.ragdolled), bit(selfState.blocking), bit(selfState.recovering),
        bit(selfState.invulnerable), norm(selfState.m1_stage, 4), norm(selfState.m1_reset_remaining_ms, 1500),
        norm(selfState.action_lock_remaining_ms, 3000), clamp(selfState.state_confidence, 0, 1),

        clamp(target.health_ratio, 0, 1),
        norm(target.speed_x, 40), norm(target.speed_z, 40), norm(target.vertical_speed, 80),
        clamp(target.facing_target_cos, -1, 1), bit(target.grounded), bit(target.airborne),
        bit(target.stunned), bit(target.ragdolled), bit(target.blocking), bit(target.recovering),
        bit(target.invulnerable), bit(target.action_active), clamp(target.action_progress, 0, 1),
        clamp(target.state_confidence, 0, 1),

        norm(relative.x, 120), norm(relative.y, 60), norm(relative.z, 120),
        norm(relative.horizontal_distance, 120), norm(relative.distance, 150),
        norm(relative.closing_speed, 60), norm(relative.lateral_speed, 60),
        norm(relative.height_advantage, 60), bit(relative.line_of_sight),
        bit(relative.in_m1_range), bit(relative.in_move_range),
        norm(relative.self_angle_error_deg, 180), norm(relative.target_angle_error_deg, 180),

        norm(combat.self_hitstun_remaining_ms, 3000), norm(combat.target_hitstun_remaining_ms, 3000),
        norm(combat.self_immunity_remaining_ms, 3000), norm(combat.target_immunity_remaining_ms, 3000),
        norm(combat.last_hit_age_ms, 5000), norm(combat.last_damage_age_ms, 5000),
        norm(combat.combo_counter, 10), norm(combat.target_combo_counter, 10),

        bit(cooldowns.move1_ready), bit(cooldowns.move2_ready), bit(cooldowns.move3_ready), bit(cooldowns.move4_ready),
        norm(cooldowns.move1_remaining_ms, 30000), norm(cooldowns.move2_remaining_ms, 30000),
        norm(cooldowns.move3_remaining_ms, 30000), norm(cooldowns.move4_remaining_ms, 30000),

        norm(network.ping_ms, 500), norm(network.jitter_ms, 200),
        clamp(history.last_action_success, -1, 1), norm(history.last_action_age_ms, 5000),
    }
    assert(#vector == FEATURE_COUNT, "feature vector shape regression")
    return vector
end

local Collector = {}
Collector.__index = Collector

function Collector.new()
    return setmetatable({
        Version = VERSION,
        SchemaVersion = SCHEMA_VERSION,
        Active = false,
        Episode = nil,
        Config = nil,
        TargetOverride = nil,
        Connections = {},
        EntityConnections = {},
        EventQueue = {},
        PendingActionRequest = nil,
        PendingExecutorResult = nil,
        PendingConfirmations = {},
        LastSelfHealth = nil,
        LastTargetHealth = nil,
        LastPing = nil,
        LastSampleMs = nil,
        LastActionMs = nil,
        LastActionSuccess = 0,
        LastEntities = nil,
    }, Collector)
end

function Collector:_connect(signal, callback, bucket)
    local connection = signal:Connect(callback)
    table.insert(bucket or self.Connections, connection)
    return connection
end

function Collector:_event(kind, payload)
    table.insert(self.EventQueue, {
        timestamp_ms = nowMs(),
        kind = kind,
        payload = payload,
    })
    while #self.EventQueue > (self.Config and self.Config.event_buffer_limit or 256) do
        table.remove(self.EventQueue, 1)
    end
end

function Collector:_attach(character, target)
    disconnectAll(self.EntityConnections)

    local function attach(model, role)
        if not model then
            return
        end
        local humanoid = getHumanoid(model)
        if humanoid then
            self:_connect(humanoid.HealthChanged, function(health)
                self:_event("health_changed", { role = role, health = health, max_health = humanoid.MaxHealth })
            end, self.EntityConnections)
            local animator = humanoid:FindFirstChildOfClass("Animator")
            if animator then
                self:_connect(animator.AnimationPlayed, function(track)
                    self:_event("animation_played", {
                        role = role,
                        animation_id = track.Animation and track.Animation.AnimationId or "",
                        priority = tostring(track.Priority),
                    })
                end, self.EntityConnections)
            end
        end

        self:_connect(model.ChildAdded, function(child)
            self:_event("marker_added", { role = role, name = child.Name, class_name = child.ClassName })
        end, self.EntityConnections)
        self:_connect(model.ChildRemoved, function(child)
            self:_event("marker_removed", { role = role, name = child.Name, class_name = child.ClassName })
        end, self.EntityConnections)

        for _, name in ipairs({ "Blocking", "Busy", "Combo", "LastHit", "FHeld", "Space" }) do
            local value = model:FindFirstChild(name)
            if value and value:IsA("ValueBase") then
                self:_connect(value.Changed, function(newValue)
                    self:_event("value_changed", { role = role, name = name, value = newValue })
                end, self.EntityConnections)
            end
        end
    end

    attach(character, "self")
    attach(target, "target")
end

function Collector:_observation(character, target, timestampMs, dtMs)
    local root = getRoot(character)
    local targetRoot = getRoot(target)
    local selfState = entityState(character, target)
    local targetState = entityState(target, character)
    local worldOffset = root and targetRoot and (targetRoot.Position - root.Position) or Vector3.zero
    local localOffset = root and root.CFrame:VectorToObjectSpace(worldOffset) or Vector3.zero
    local horizontalDistance = Vector3.new(worldOffset.X, 0, worldOffset.Z).Magnitude
    local distance = worldOffset.Magnitude
    local selfVelocity = root and root.AssemblyLinearVelocity or Vector3.zero
    local targetVelocity = targetRoot and targetRoot.AssemblyLinearVelocity or Vector3.zero
    local relativeVelocity = targetVelocity - selfVelocity
    local forward = horizontalDistance > 0.001 and Vector3.new(worldOffset.X, 0, worldOffset.Z).Unit or Vector3.zero
    local lateral = Vector3.new(-forward.Z, 0, forward.X)

    local moves = { moveHud(1), moveHud(2), moveHud(3), moveHud(4) }
    local ping = pingMs()
    local jitter = self.LastPing and math.abs(ping - self.LastPing) or 0
    self.LastPing = ping

    local selfCombo = tonumber(readValue(character, "Combo", 0)) or 0
    local targetCombo = tonumber(readValue(target, "Combo", 0)) or 0

    local observation = {
        schema_version = SCHEMA_VERSION,
        episode_id = self.Episode.header.episode_id,
        step_index = #self.Episode.steps,
        timestamp_ms = timestampMs,
        dt_ms = dtMs,
        self_state = selfState,
        target_state = targetState,
        relative = {
            x = localOffset.X, y = localOffset.Y, z = localOffset.Z,
            horizontal_distance = horizontalDistance,
            distance = distance,
            closing_speed = -Vector3.new(relativeVelocity.X, 0, relativeVelocity.Z):Dot(forward),
            lateral_speed = Vector3.new(relativeVelocity.X, 0, relativeVelocity.Z):Dot(lateral),
            height_advantage = -worldOffset.Y,
            line_of_sight = lineOfSight(character, target, root, targetRoot),
            in_m1_range = distance <= 5.65,
            in_move_range = distance <= 30,
            self_angle_error_deg = root and angleError(root.CFrame.LookVector, worldOffset) or 0,
            target_angle_error_deg = targetRoot and angleError(targetRoot.CFrame.LookVector, -worldOffset) or 0,
        },
        combat = {
            self_hitstun_remaining_ms = 0,
            target_hitstun_remaining_ms = 0,
            self_immunity_remaining_ms = 0,
            target_immunity_remaining_ms = 0,
            last_hit_age_ms = 5000,
            last_damage_age_ms = 5000,
            combo_counter = selfCombo,
            target_combo_counter = targetCombo,
            confidence = 0.45,
        },
        cooldowns = {
            move1_ready = moves[1].ready, move2_ready = moves[2].ready,
            move3_ready = moves[3].ready, move4_ready = moves[4].ready,
            move1_remaining_ms = moves[1].fill_ratio * 30000,
            move2_remaining_ms = moves[2].fill_ratio * 30000,
            move3_remaining_ms = moves[3].fill_ratio * 30000,
            move4_remaining_ms = moves[4].fill_ratio * 30000,
            hud = moves,
            source = "hud_cooldown_fill",
            remaining_time_is_estimate = true,
            confidence = 0.9,
        },
        network = { ping_ms = ping, jitter_ms = jitter, source = "Stats.Network.ServerStatsItem.Data Ping" },
        history = {
            last_action_success = self.LastActionSuccess,
            last_action_age_ms = self.LastActionMs and math.max(0, timestampMs - self.LastActionMs) or 5000,
        },
        feature_vector = {},
    }
    observation.feature_vector = featureVector(observation)
    return observation
end

function Collector:_mask(observation)
    local selfState = observation.self_state
    local cooldowns = observation.cooldowns
    local alive = selfState.health > 0
    local incapacitated = selfState.stunned or selfState.ragdolled or selfState.recovering
    local canAct = alive and not incapacitated and not selfState.action_active
    local slots = {
        canAct and cooldowns.move1_ready,
        canAct and cooldowns.move2_ready,
        canAct and cooldowns.move3_ready,
        canAct and cooldowns.move4_ready,
    }
    return {
        schema_version = SCHEMA_VERSION,
        commands = {
            none = true,
            m1 = canAct,
            block_start = canAct and not selfState.blocking,
            block_stop = alive and selfState.blocking,
            dodge = canAct,
            jump = alive and not selfState.stunned and not selfState.ragdolled,
            move = slots[1] or slots[2] or slots[3] or slots[4],
            transform = false,
            reset_spacing = alive and not selfState.ragdolled,
        },
        move_slots = slots,
        reasons = {
            can_act = canAct and "direct_state_allows" or "incapacitated_or_busy",
            transform = "not_instrumented",
            cooldown_source = cooldowns.source,
        },
        confidence = 0.82,
    }
end

function Collector:_sample()
    if not self.Active or not self.Episode then
        return
    end
    if #self.Episode.steps >= self.Config.max_steps then
        self:Stop("max_steps")
        return
    end

    local character = LocalPlayer and LocalPlayer.Character
    local target = resolveTarget(character, self.TargetOverride)
    if not character or not target then
        self:_event("sample_skipped", { reason = not character and "missing_character" or "missing_target" })
        return
    end

    local entityKey = character:GetDebugId() .. ":" .. target:GetDebugId()
    if self.LastEntities ~= entityKey then
        self.LastEntities = entityKey
        self.Episode.header.character = character.Name
        self.Episode.header.target = target.Name
        self.LastSelfHealth = nil
        self.LastTargetHealth = nil
        self:_attach(character, target)
    end

    local timestamp = nowMs()
    local dt = self.LastSampleMs and clamp(timestamp - self.LastSampleMs, 0, 1000) or 0
    self.LastSampleMs = timestamp
    local observation = self:_observation(character, target, timestamp, dt)
    local events = self.EventQueue
    self.EventQueue = {}
    local confirmations = self.PendingConfirmations
    self.PendingConfirmations = {}
    local damageDealt = 0
    local damageReceived = 0

    if self.LastSelfHealth and observation.self_state.health < self.LastSelfHealth then
        damageReceived = self.LastSelfHealth - observation.self_state.health
        table.insert(confirmations, {
            kind = "damage_received", timestamp_ms = timestamp, amount = damageReceived,
            source = "humanoid_health_delta", confidence = 1,
        })
    end
    if self.LastTargetHealth and observation.target_state.health < self.LastTargetHealth then
        damageDealt = self.LastTargetHealth - observation.target_state.health
        table.insert(confirmations, {
            kind = "damage_dealt", timestamp_ms = timestamp, amount = damageDealt,
            source = "target_humanoid_health_delta", confidence = 1,
        })
    end
    self.LastSelfHealth = observation.self_state.health
    self.LastTargetHealth = observation.target_state.health

    for _, event in ipairs(events) do
        if event.kind == "animation_played" or event.kind == "marker_added"
            or event.kind == "marker_removed" or event.kind == "value_changed" then
            table.insert(confirmations, {
                kind = event.kind,
                timestamp_ms = event.timestamp_ms,
                source = "direct_signal",
                confidence = 1,
                payload = event.payload,
            })
        end
    end

    local actionRequest = self.PendingActionRequest
    local executorResult = self.PendingExecutorResult or {
        status = actionRequest and "unobserved" or "no_request",
        source = "collector_observation_only",
    }
    if actionRequest then
        self.LastActionMs = timestamp
    end
    if executorResult.status == "confirmed" then
        self.LastActionSuccess = 1
    elseif executorResult.status == "rejected" then
        self.LastActionSuccess = -1
    end

    table.insert(self.Episode.steps, {
        observation = observation,
        action_mask = self:_mask(observation),
        action_request = actionRequest,
        executor_result = executorResult,
        confirmations = confirmations,
        rewards = {
            damage_dealt = damageDealt,
            damage_received = damageReceived,
            invalid_request = executorResult.status == "rejected" and 1 or 0,
        },
        raw_events = events,
    })

    self.PendingActionRequest = nil
    self.PendingExecutorResult = nil
end

function Collector:Start(options)
    if self.Active then
        self:Stop("restarted")
    end
    options = options or {}
    disconnectAll(self.Connections)
    disconnectAll(self.EntityConnections)

    local sampleHz = clamp(options.sample_hz or 15, 1, 60)
    local maxSteps = math.floor(clamp(options.max_steps or 18000, 1, 100000))
    local started = nowMs()
    local character = LocalPlayer and LocalPlayer.Character
    local target = resolveTarget(character, options.target or self.TargetOverride)

    self.Config = {
        sample_hz = sampleHz,
        interval = 1 / sampleHz,
        max_steps = maxSteps,
        event_buffer_limit = math.floor(clamp(options.event_buffer_limit or 256, 32, 4096)),
        export_directory = options.export_directory or "aba_deep_learning/episodes",
    }
    self.TargetOverride = options.target or self.TargetOverride
    self.Episode = {
        schema_version = SCHEMA_VERSION,
        header = {
            episode_id = options.episode_id or HttpService:GenerateGUID(false),
            agent_build = "0.2.0-collector",
            collector_version = VERSION,
            stream_format = STREAM_FORMAT,
            started_at_ms = started,
            place_id = game.PlaceId,
            place_version = game.PlaceVersion,
            job_id = game.JobId,
            player = LocalPlayer and LocalPlayer.Name or nil,
            character = character and character.Name or nil,
            target = target and target.Name or nil,
            sample_hz = sampleHz,
            max_steps = maxSteps,
            observation_only = true,
        },
        steps = {},
        terminal = nil,
    }

    self.EventQueue = {}
    self.PendingActionRequest = nil
    self.PendingExecutorResult = nil
    self.PendingConfirmations = {}
    self.LastSelfHealth = nil
    self.LastTargetHealth = nil
    self.LastPing = nil
    self.LastSampleMs = nil
    self.LastActionMs = nil
    self.LastActionSuccess = 0
    self.LastEntities = nil
    self.Active = true

    self:_attach(character, target)
    self:_connect(UserInputService.InputBegan, function(input, processed)
        self:_event("input_began", {
            key_code = tostring(input.KeyCode),
            user_input_type = tostring(input.UserInputType),
            processed = processed,
        })
    end)
    self:_connect(UserInputService.InputEnded, function(input, processed)
        self:_event("input_ended", {
            key_code = tostring(input.KeyCode),
            user_input_type = tostring(input.UserInputType),
            processed = processed,
        })
    end)

    local accumulator = 0
    self:_connect(RunService.Heartbeat, function(dt)
        accumulator += dt
        if accumulator >= self.Config.interval then
            accumulator %= self.Config.interval
            local ok, err = pcall(function()
                self:_sample()
            end)
            if not ok then
                self:_event("collector_error", { message = tostring(err) })
            end
        end
    end)

    self:_event("episode_started", { episode_id = self.Episode.header.episode_id, sample_hz = sampleHz })
    return self.Episode.header.episode_id
end

function Collector:Stop(reason)
    if not self.Episode then
        return nil
    end
    self.Active = false
    disconnectAll(self.Connections)
    disconnectAll(self.EntityConnections)
    if not self.Episode.terminal then
        local ended = nowMs()
        self.Episode.terminal = {
            reason = reason or "stopped",
            ended_at_ms = ended,
            duration_ms = math.max(0, ended - self.Episode.header.started_at_ms),
            steps = #self.Episode.steps,
            winner = nil,
            truncated = #self.Episode.steps >= self.Config.max_steps,
        }
    end
    return self.Episode
end

function Collector:SetTarget(target)
    assert(target == nil or typeof(target) == "Instance", "target must be an Instance or nil")
    self.TargetOverride = target
end

function Collector:RecordActionRequest(request, source)
    assert(type(request) == "table", "request must be a table")
    request.schema_version = request.schema_version or SCHEMA_VERSION
    request.policy_id = request.policy_id or tostring(source or "external")
    request.model_version = request.model_version or "external"
    self.PendingActionRequest = request
    self:_event("action_request_recorded", { source = source or "external", request = request })
end

function Collector:RecordExecutorResult(result)
    assert(type(result) == "table", "result must be a table")
    self.PendingExecutorResult = result
    self:_event("executor_result_recorded", result)
end

function Collector:Confirm(kind, payload)
    table.insert(self.PendingConfirmations, {
        kind = tostring(kind),
        timestamp_ms = nowMs(),
        source = "external_confirmation",
        confidence = 1,
        payload = payload,
    })
end

function Collector:GetEpisode()
    return self.Episode
end

function Collector:GetSummary()
    local episode = self.Episode
    if not episode then
        return { active = self.Active, version = VERSION, steps = 0 }
    end
    local damageDealt = 0
    local damageReceived = 0
    local requests = 0
    local confirmations = 0
    local rawEvents = 0
    for _, step in ipairs(episode.steps) do
        damageDealt += tonumber(step.rewards.damage_dealt) or 0
        damageReceived += tonumber(step.rewards.damage_received) or 0
        requests += step.action_request and 1 or 0
        confirmations += #step.confirmations
        rawEvents += #step.raw_events
    end
    return {
        active = self.Active,
        version = VERSION,
        schema_version = SCHEMA_VERSION,
        episode_id = episode.header.episode_id,
        steps = #episode.steps,
        target = episode.header.target,
        duration_ms = episode.terminal and episode.terminal.duration_ms
            or math.max(0, nowMs() - episode.header.started_at_ms),
        damage_dealt = damageDealt,
        damage_received = damageReceived,
        action_requests = requests,
        confirmations = confirmations,
        raw_events = rawEvents,
        terminal_reason = episode.terminal and episode.terminal.reason or nil,
    }
end

function Collector:Validate()
    local episode = self.Episode
    if not episode then
        return false, "no episode"
    end
    local previous = -1
    for position, step in ipairs(episode.steps) do
        local observation = step.observation
        if observation.schema_version ~= SCHEMA_VERSION then
            return false, "step " .. position .. " schema mismatch"
        end
        if observation.episode_id ~= episode.header.episode_id then
            return false, "step " .. position .. " episode mismatch"
        end
        if observation.step_index <= previous then
            return false, "step indices are not increasing"
        end
        previous = observation.step_index
        if type(observation.feature_vector) ~= "table" or #observation.feature_vector ~= FEATURE_COUNT then
            return false, "step " .. position .. " feature-vector shape mismatch"
        end
    end
    return true
end

function Collector:Export(path)
    local episode = self:Stop("exported")
    assert(episode, "no episode to export")
    local directory = self.Config.export_directory
    if type(makefolder) == "function" and type(isfolder) == "function" then
        local current = ""
        for segment in string.gmatch(directory, "[^/\\]+") do
            current = current == "" and segment or (current .. "/" .. segment)
            if not isfolder(current) then
                makefolder(current)
            end
        end
    end
    path = path or (directory .. "/" .. episode.header.episode_id .. ".jsonl")
    local lines = {
        HttpService:JSONEncode({ record_type = "header", schema_version = SCHEMA_VERSION, header = episode.header }),
    }
    for _, step in ipairs(episode.steps) do
        table.insert(lines, HttpService:JSONEncode({ record_type = "step", schema_version = SCHEMA_VERSION, step = step }))
    end
    table.insert(lines, HttpService:JSONEncode({ record_type = "terminal", schema_version = SCHEMA_VERSION, terminal = episode.terminal }))
    local payload = table.concat(lines, "\n") .. "\n"
    assert(type(writefile) == "function", "writefile is unavailable")
    writefile(path, payload)
    return { path = path, bytes = #payload, lines = #lines, steps = #episode.steps }
end

function Collector:Destroy()
    self:Stop("destroyed")
    if getgenv().ABA_NEURAL_COLLECTOR == self then
        getgenv().ABA_NEURAL_COLLECTOR = nil
    end
end

local previous = getgenv().ABA_NEURAL_COLLECTOR
if previous and type(previous.Destroy) == "function" then
    pcall(function()
        previous:Destroy()
    end)
end

local collector = Collector.new()
getgenv().ABA_NEURAL_COLLECTOR = collector
print("[ABA Neural Collector] installed", VERSION)
return collector
